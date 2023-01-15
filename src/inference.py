import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from transformer import make_transformer
from ferminet import make_ferminet 
#from energy import make_energy, make_free_energy
from flow import make_flow 
import checkpoint
import utils 
import matplotlib.pyplot as plt 

import os
import time

import argparse
parser = argparse.ArgumentParser(description='')

group = parser.add_argument_group('learning parameters')
group.add_argument('--batchsize', type=int, default=4096, help='')

group = parser.add_argument_group('filesystem')
group.add_argument("--restore_path", default=None, help="checkpoint path or file")

group = parser.add_argument_group("datasets")
group.add_argument("--dataset", default="../data/LJSystem_npy/liquid/traj_N32_rho0.7_T1.0.npy",help="The path to training dataset")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--transformer", action="store_true", help="Use transformer")
group.add_argument("--ferminet", action="store_true", help="Use ferminet")

group = parser.add_argument_group("transformer parameters")
group.add_argument("--nlayers", type=int, default=4, help="The number of layers")
group.add_argument("--nheads", type=int, default=8, help="")
group.add_argument("--keysize", type=int, default=16, help="")

group = parser.add_argument_group("ferminet parameters")
group.add_argument("--depth", type=int, default=3, help="The number of layers")
group.add_argument("--h1size", type=int, default=32, help="")
group.add_argument("--h2size", type=int, default=16, help="")


group = parser.add_argument_group('physics parameters')
group.add_argument('--n', type=int, default=6, help='The number of particles')
group.add_argument('--dim', type=int, default=2, help='The dimensions of the system')
group.add_argument('--beta', type=float, default=10.0, help='inverse temperature')

args = parser.parse_args()

key = jax.random.PRNGKey(42)

print("\n========== Prepare training dataset ==========")

if os.path.isfile(args.dataset):
    data = jnp.load(args.dataset)
    X1 = data
    datasize, n, dim = X1.shape[0], X1.shape[1], X1.shape[2]
    X1 = X1.reshape(datasize, n*dim)
    L = 12.225024745980599
    print (jnp.min(X1), jnp.max(X1))
    X1 -= L * jnp.floor(X1/L)
    print("Load dataset: %s" % args.dataset)
else:
    raise ValueError("what dataset ?")
####################################################################################


if args.transformer:
    print("\n========== Construct transformer ==========")
    params, vec_field_net = make_transformer(key, n, dim, args.nheads, args.nlayers, args.keysize, L)
    modelname = "transformer_l_%d_h_%d_k_%d" % (args.nlayers, args.nheads, args.keysize)
elif args.ferminet:
    print("\n========== Construct ferminet ==========")
    params, vec_field_net = make_ferminet(key, n, dim, args.depth, args.h1size, args.h2size, L)
    modelname = "ferminet_d_%d_h1_%d_h2_%d" % (args.depth, args.h1size, args.h2size)
else:
    raise ValueError("what model ?")

raveled_params, _ = ravel_pytree(params)
print("# of params: %d" % raveled_params.size)

key, subkey = jax.random.split(key)
sampler, sampler_with_logp = make_flow(vec_field_net, n*dim, L)
#energy_fn = make_energy(args.n, args.dim)
#free_energy_fn = make_free_energy(energy_fn, sampler, args.n, args.dim, args.beta)

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
start = time.time()
key, subkey = jax.random.split(key)
x = sampler(subkey, params, args.batchsize)
print ('sample shape', x.shape)

rdf_data = utils.get_gr(X1.reshape(-1, n, dim), L)
print ('data shape', X1.shape)
plt.plot(rdf_data[0], rdf_data[1], linestyle='-', c='blue', label='data')
for t in range(x.shape[1]):
    rdf_model = utils.get_gr(x[:, t, :].reshape(-1, n, dim), L)
    plt.plot(rdf_model[0], rdf_model[1], linestyle='-', 
             label='model_%g'%(t/(x.shape[1]-1)),
             )
             #alpha= (t/(x.shape[1]-1) + 0.1)/1.1, c='red'
             #)
plt.legend()
plt.show()
import sys 
sys.exit(1)

#fe, fe_err, x, vfe, vfe_err = free_energy_fn(subkey, params, args.batchsize)
end = time.time()
running_time = end - start
#print('free energy using trained model: %f ± %f' %(fe, fe_err))
#print('variational free energy using trained model: %f ± %f' %(vfe, vfe_err))
#print('importance sampling time: %.5f sec' %running_time)
