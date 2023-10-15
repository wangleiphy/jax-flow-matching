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
from train import train
from hungarian import matching
import utils

import argparse
import os

####################################################################################

parser = argparse.ArgumentParser(description="")

group = parser.add_argument_group("learning parameters")
group.add_argument("--epochs", type=int, default=100000, help="Epochs for training")
group.add_argument("--batchsize", type=int, default=1000, help="")
group.add_argument("--lr", type=float, default=1e-3, help="learning rate")
group.add_argument("--fmax", type=float, default=0, help="clip force, 0 means we do not use force ref.")
group.add_argument("--folder", default="../data/", help="The folder to save data")

group = parser.add_argument_group("datasets")
group.add_argument("--X0", default="/data/zhangqidata/TestHelium3Flow/Helium3FreeFermions_n_14/epoch_000400.pkl",help="The path to training dataset")
group.add_argument("--X1", default="/data/zhangqidata/TestHelium3Flow/Helium3Jastrow_n_14/epoch_004000.pkl",help="The path to training dataset")
group.add_argument("--permute", action="store_true", help="permute particle")

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

if os.path.isfile(args.X0) and os.path.isfile(args.X1):
    dataname = os.path.splitext(os.path.basename(args.X0))[0]+'_' +\
               os.path.splitext(os.path.basename(args.X1))[0]
    print (dataname)

    X1, n, dim, L, _ = utils.loaddata(args.X1)
    #X0, _, _, _, _ = utils.loaddata(args.X0)
    
    key, subkey = jax.random.split(key)
    X0 = jax.random.uniform(subkey, X1.shape, minval=0, maxval=L)
    
    key, subkey1, subkey2 = jax.random.split(key, 3)
    X0 = jax.random.permutation(subkey1, X0)
    X1 = jax.random.permutation(subkey2, X1)

    datasize = min(X0.shape[0], X1.shape[0])
    assert (datasize % args.batchsize == 0)

    X0 = X0[:datasize].reshape(-1, n*dim)
    X1 = X1[:datasize].reshape(-1, n*dim)
    
    def dist_fn(x0, x1):
        rij = jnp.reshape(x0, (n, dim)) - jnp.reshape(x1, (n, dim))
        rij = rij - L*jnp.rint(rij/L)
        r = jnp.linalg.norm(rij, axis=-1) #(n, )
        return r*r

    print ('total dist:', jax.vmap(dist_fn)(X0, X1).sum())
    if args.permute:
        dataname += '_permute'
        from tqdm import tqdm
        for b in tqdm(range(len(X0))):
            _, x1 = matching(X0[b].reshape(n, dim), X1[b].reshape(n, dim), L)
            X1 = X1.at[b].set( jnp.reshape(x1, (n*dim, )))
        print ('total dist after permute:', jax.vmap(dist_fn)(X0, X1).sum())

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
    modelname = "hollownet_nh_%d_k_%d_h1_%d_h2_%d_l_%d" % (args.nheads, args.keysize, args.h1size, args.h2size, args.nlayers)
else:
    raise ValueError("what model ?")

raveled_params, _ = ravel_pytree(params)
print("# of params: %d" % raveled_params.size)

sampler, sampler_with_logp = make_flow(vec_field_net, div_fn, X0, X1)

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

params = train(key, value_and_grad, args.epochs, args.batchsize, params, X0, X1, args.lr, path, L)

####################################################################################
