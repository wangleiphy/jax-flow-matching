import jax
import jax.numpy as jnp

from transformer import make_transformer
from loss import make_loss
from train import train

import argparse
import time
import os

jax.config.update("jax_enable_x64", True)
rng = jax.random.PRNGKey(42)

####################################################################################

parser = argparse.ArgumentParser(description="")

group = parser.add_argument_group("learning parameters")
group.add_argument("--epochs", type=int, default=1000, help="Epochs for training")
group.add_argument("--batchsize", type=int, default=4096, help="")
group.add_argument("--lr", type=float, default=1e-3, help="learning rate")
group.add_argument("--folder", default="../data/", help="The folder to save data")

group = parser.add_argument_group("datasets")
group.add_argument("--dataset", default="../datasets/data.npz",help="The path to training dataset")

group = parser.add_argument_group("network parameters")
group.add_argument("--nlayers", type=int, default=4, help="The number of layers")
group.add_argument("--nheads", type=int, default=8, help="")
group.add_argument("--keysize", type=int, default=16, help="")

args = parser.parse_args()

####################################################################################

print("\n========== Prepare training dataset ==========")

if os.path.isfile(args.dataset):
    data = jnp.load(args.dataset)
    X1 = data["X1"]
    datasize, n, dim = X1.shape[0], X1.shape[1], X1.shape[2]
    assert (datasize % args.batchsize == 0)
    L = data["L"] 
    print("Load dataset: %s" % args.dataset)
else:
    raise ValueError("what dataset ?")
####################################################################################

init_rng, rng = jax.random.split(rng)
print("\n========== Construct transformer network ==========")
params, vec_field_net = make_transformer(init_rng, n, dim, args.nheads, args.nlayers, args.keysize, L)
modelname = "transformer_nl_%d_nh_%d_nk_%d" % (args.nlayers, args.nheads, args.keysize)

"""initializing the loss function"""
loss = make_loss(vec_field_net)
value_and_grad = jax.value_and_grad(loss)

####################################################################################

print("\n========== Prepare logs ==========")

path = args.folder + "n_%d_dim_%d_%g_lr_%g" % (args.n, args.dim, args.lr) \
                    + "_" + modelname
os.makedirs(path, exist_ok=True)
print("Create directory: %s" % path)

####################################################################################

print("\n========== Train ==========")

start = time.time()
params = train(rng, value_and_grad, args.epoch, args.batchsize, params, X1, args.lr, path)
end = time.time()
running_time = end - start
print("training time: %.5f sec" %running_time)

####################################################################################
