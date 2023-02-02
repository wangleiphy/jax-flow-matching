import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from transformer import make_transformer
from ferminet import make_ferminet 
from hollow import make_hollow_net 
from energy import make_energy
from loss import make_loss
from train import train
import utils

import argparse
import time
import os

####################################################################################

parser = argparse.ArgumentParser(description="")

group = parser.add_argument_group("learning parameters")
group.add_argument("--epochs", type=int, default=100000, help="Epochs for training")
group.add_argument("--batchsize", type=int, default=1000, help="")
group.add_argument("--lr", type=float, default=1e-3, help="learning rate")
group.add_argument("--fmax", type=float, default=1e5, help="clip force")
group.add_argument("--folder", default="../data/", help="The folder to save data")

group = parser.add_argument_group("datasets")
group.add_argument("--dataset", default="../data/LJSystem_npz/liquid/traj_N32_rho0.7_T1.0.npz",help="The path to training dataset")

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

args = parser.parse_args()

####################################################################################

key = jax.random.PRNGKey(42)

print("\n========== Prepare training dataset ==========")

if os.path.isfile(args.dataset):
    X1, n, dim, L, _ = utils.loaddata(args.dataset)
    assert (X1.shape[0]% args.batchsize == 0)
    print("Load dataset: %s" % args.dataset)
    dataname = os.path.splitext(os.path.basename(args.dataset))[0]
else:
    raise ValueError("what dataset ?")
####################################################################################

key, subkey = jax.random.split(key)

if args.transformer:
    print("\n========== Construct transformer ==========")
    params, vec_field_net, _ = make_transformer(subkey, n, dim, args.nheads, args.nlayers, args.keysize, L)
    modelname = "transformer_l_%d_h_%d_k_%d" % (args.nlayers, args.nheads, args.keysize)
elif args.ferminet:
    print("\n========== Construct ferminet ==========")
    params, vec_field_net, _ = make_ferminet(subkey, n, dim, args.nlayers, args.h1size, args.h2size, L, args.fmax)
    modelname = "ferminet_l_%d_h1_%d_h2_%d" % (args.nlayers, args.h1size, args.h2size)
elif args.hollow:
    print("\n========== Construct hollownet ==========")
    params, vec_field_net, _ = make_hollow_net(subkey, n, dim, L, args.nheads, args.keysize, args.h1size, args.h2size, args.nlayers)
    modelname = "hollownet_nh_%d_k_%d_h1_%d_h2_%d_l_%d" % (args.nheads, args.keysize, args.h1size, args.h2size, args.nlayers)
else:
    raise ValueError("what model ?")

raveled_params, _ = ravel_pytree(params)
print("# of params: %d" % raveled_params.size)

loss = make_loss(vec_field_net, L)
value_and_grad = jax.value_and_grad(loss)
####################################################################################

print("\n========== Prepare logs ==========")

path = args.folder + dataname \
                   + "_" + modelname \
                   + "_lr_%g_fmax_%g_bs_%g" % (args.lr, args.fmax, args.batchsize) 
os.makedirs(path, exist_ok=True)
print("Create directory: %s" % path)

####################################################################################

print("\n========== Train ==========")

start = time.time()
params = train(key, value_and_grad, args.epochs, args.batchsize, params, X1, args.lr, path, L)
end = time.time()
running_time = end - start
print("training time: %.5f sec" %running_time)

####################################################################################
