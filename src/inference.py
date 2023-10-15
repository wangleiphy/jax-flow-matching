import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from transformer import make_transformer
from ferminet import make_ferminet 
from hollow import make_hollow_net 

from flow import make_flow 
from loss import make_loss
import checkpoint
import utils 
import matplotlib.pyplot as plt 

import sys
import os
import time

import argparse
parser = argparse.ArgumentParser(description='')

group = parser.add_argument_group('learning parameters')
group.add_argument('--batchsize', type=int, default=4096, help='')

group = parser.add_argument_group('filesystem')
group.add_argument("--restore_path", default=None, help="checkpoint path or file")

group = parser.add_argument_group("datasets")
group.add_argument("--X0", default="../data/LJTraj_WCA/liquid/traj_N32_rho0.7_T1.0.npz",help="The path to training dataset")
group.add_argument("--X1", default="../data/LJSystem_npz/liquid/traj_N32_rho0.7_T1.0.npz",help="The path to training dataset")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--hollow", action="store_true", help="Use hollownet")
group.add_argument("--transformer", action="store_true", help="Use transformer")
group.add_argument("--ferminet", action="store_true", help="Use ferminet")

group = parser.add_argument_group("network parameters")
group.add_argument("--nlayers", type=int, default=2, help="The number of layers")
group.add_argument("--nheads", type=int, default=8, help="")
group.add_argument("--keysize", type=int, default=16, help="")
group.add_argument("--h1size", type=int, default=32, help="")
group.add_argument("--h2size", type=int, default=32, help="")
group.add_argument("--fmax", type=float, default=0, help="clip force")

args = parser.parse_args()

key = jax.random.PRNGKey(42)

print("\n========== Prepare training dataset ==========")

if os.path.isfile(args.X0) and os.path.isfile(args.X1):
    X1, n, dim, L, _ = utils.loaddata(args.X1)
    X0, _, _, _, _ = utils.loaddata(args.X0)

    X0 = X0.reshape(-1, n*dim)
    X1 = X1.reshape(-1, n*dim)
else:
    raise ValueError("what dataset ?")
####################################################################################
key, subkey = jax.random.split(key)

if args.transformer:
    print("\n========== Construct transformer ==========")
    params, vec_field_net, div_fn = make_transformer(subkey, n, dim, args.nheads, args.nlayers, args.keysize, L)
    modelname = "transformer_l_%d_h_%d_k_%d" % (args.nlayers, args.nheads, args.keysize)
elif args.ferminet:
    print("\n========== Construct ferminet ==========")
    params, vec_field_net, div_fn = make_ferminet(subkey, n, dim, args.nlayers, args.h1size, args.h2size, L, args.fmax)
    modelname = "ferminet_l_%d_h1_%d_h2_%d" % (args.nlayers, args.h1size, args.h2size)
elif args.hollow:
    print("\n========== Construct hollownet ==========")
    params, vec_field_net, div_fn = make_hollow_net(subkey, n, dim, L, args.nheads, args.keysize, args.h1size, args.h2size, args.nlayers)
    modelname = "hollownet_n_%d_k_%d_h1_%d_h2_%d_l_%d" % (args.nheads, args.keysize, args.h1size, args.h2size, args.nlayers)
else:
    raise ValueError("what model ?")

raveled_params, _ = ravel_pytree(params)
print("# of params: %d" % raveled_params.size)

sampler, sampler_with_logp = make_flow(vec_field_net, div_fn, X0, X1)

print("\n========== Prepare logs ==========")

folder = os.path.dirname(args.restore_path)
ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path)

print ('folder:', folder)
if ckpt_filename is not None:
    print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
    ckpt = checkpoint.load_data(ckpt_filename)
    params = ckpt["params"]
else:
    raise ValueError("no checkpoint found")

print("\n========== Start inference ==========")
'''
key, key_x0, key_t = jax.random.split(key, 3)
x1 = X1[:args.batchsize]
x0 = jax.random.uniform(key_x0, x1.shape, minval=0, maxval=L)
t = jax.random.uniform(key_t, (args.batchsize,))
_, loss_fn = make_loss(vec_field_net, L)
loss = loss_fn(params, x0, x1, t)
print (t.shape, loss.shape)
plt.plot(t, loss, 'o')
plt.show()
'''

key, subkey = jax.random.split(key)
x = sampler(subkey, params, args.batchsize, True)
print ('sample shape', x.shape)

rdf_data = utils.get_gr(X1.reshape(-1, n, dim), L)
print ('data shape', X1.shape)
plt.plot(rdf_data[0], rdf_data[1], linestyle='-', c='blue', label='data')

for t in [0,  x.shape[1]-1]:
    rdf_model = utils.get_gr(x[:, t, :].reshape(-1, n, dim), L)
    plt.plot(rdf_model[0], rdf_model[1], linestyle='-', 
             label='model@t=%g'%(t/(x.shape[1]-1)),
             alpha= (t/(x.shape[1]-1) + 0.1)/1.1, 
             c='red', 
             )
plt.title('epoch=%g'%epoch_finished)
plt.legend()
plt.show()
