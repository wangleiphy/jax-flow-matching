import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from transformer import make_transformer
from ferminet import make_ferminet 
from loss import make_loss
from train import train

import argparse
import time
import os

####################################################################################

parser = argparse.ArgumentParser(description="")

group = parser.add_argument_group("learning parameters")
group.add_argument("--epochs", type=int, default=10000, help="Epochs for training")
group.add_argument("--batchsize", type=int, default=1000, help="")
group.add_argument("--lr", type=float, default=1e-3, help="learning rate")
group.add_argument("--folder", default="../data/", help="The folder to save data")

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

args = parser.parse_args()

####################################################################################

key = jax.random.PRNGKey(42)

print("\n========== Prepare training dataset ==========")

if os.path.isfile(args.dataset):
    data = jnp.load(args.dataset)
    X1 = data
    datasize, n, dim = X1.shape[0], X1.shape[1], X1.shape[2]
    X1 = X1.reshape(datasize, n*dim)
    assert (datasize % args.batchsize == 0)
    L = 12.225024745980599
    print (jnp.min(X1), jnp.max(X1))
    X1 -= L * jnp.floor(X1/L)
    print("Load dataset: %s" % args.dataset)
else:
    raise ValueError("what dataset ?")
####################################################################################

key, subkey = jax.random.split(key)

if args.transformer:
    print("\n========== Construct transformer ==========")
    params, vec_field_net = make_transformer(subkey, n, dim, args.nheads, args.nlayers, args.keysize, L)
    modelname = "transformer_l_%d_h_%d_k_%d" % (args.nlayers, args.nheads, args.keysize)
elif args.ferminet:
    print("\n========== Construct ferminet ==========")
    params, vec_field_net = make_ferminet(subkey, n, dim, args.depth, args.h1size, args.h2size, L)
    modelname = "ferminet_d_%d_h1_%d_h2_%d" % (args.depth, args.h1size, args.h2size)
else:
    raise ValueError("what model ?")

raveled_params, _ = ravel_pytree(params)
print("# of params: %d" % raveled_params.size)

loss = make_loss(vec_field_net, L)
value_and_grad = jax.value_and_grad(loss)
####################################################################################

print("\n========== Prepare logs ==========")

path = args.folder + "n_%d_dim_%g_lr_%g" % (n, dim, args.lr) \
                    + "_" + modelname
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
