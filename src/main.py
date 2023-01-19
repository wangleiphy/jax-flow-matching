import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from data import sample_target
from net import make_hamiltonian_net, make_backflow, make_transformer
from loss import make_loss
from energy import make_energy
from train import train
import checkpoint

import os
import time

if __name__ == '__main__':
    key = jax.random.PRNGKey(42)

    import argparse
    parser = argparse.ArgumentParser(description='')

    group = parser.add_argument_group('learning parameters')
    group.add_argument('--epochs', type=int, default=1000, help='')
    group.add_argument('--batchsize', type=int, default=4096, help='')
    group.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    group.add_argument("--folder", default="../data/", help="the folder to save data")
    group.add_argument("--restore_path", default=None, help="checkpoint path or file")

    group = parser.add_argument_group('datasets')
    group.add_argument('--datasize', type=int, default=102400, help='')

    group = parser.add_argument_group('network parameters')
    group.add_argument('--nhiddens', type=int, default=512, help='The channels in a middle layer')
    group.add_argument('--nlayers', type=int, default=4, help='The number of layers')
    group.add_argument('--nheads', type=int, default=8, help='')
    group.add_argument('--keysize', type=int, default=16, help='')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--backflow', action='store_true', help='Use backflow')
    group.add_argument('--transformer', action='store_true', help='Use transformer')
    group.add_argument('--mlp', action='store_true', help='mlp')

    group = parser.add_argument_group('physics parameters')
    group.add_argument('--n', type=int, default=6, help='The number of particles')
    group.add_argument('--dim', type=int, default=2, help='The dimensions of the system')
    group.add_argument('--beta', type=float, default=10.0, help='inverse temperature')

    args = parser.parse_args()

    print("\n========== Generate training dataset ==========")

    start = time.time()
    energy_fn, potential_fn = make_energy(args.n, args.dim)
    logp_fn = lambda q: -args.beta * potential_fn(q)
    X1 = sample_target(key, args.datasize, args.n, args.dim, logp_fn)
    end = time.time()
    running_time = end - start
    print('training set sampling time: %.5f sec' %running_time)

    print("\n========== Build networks ==========")
    key, subkey = jax.random.split(key)
    if args.backflow:
        print ('construct backflow network')
        params, hamiltonian_net = make_backflow(subkey, args.n, args.dim, [args.nhiddens]*args.nlayers)
        modelname = 'backflow_nl_%d_nh_%d'%(args.nlayers, args.nhiddens)
    elif args.transformer:
        print ('construct transformer network')
        params, hamiltonian_net = make_transformer(subkey, args.n, args.dim, args.nheads, args.nlayers, args.keysize)
        modelname = 'transformer_nl_%d_nh_%d_nk_%d'%(args.nlayers, args.nheads, args.keysize)
    elif args.mlp:
        print ('construct mlp network')
        params, hamiltonian_net = make_hamiltonian_net(subkey, args.n, args.dim, ch=args.nhiddens, num_layers=args.nlayers)
        modelname = 'mlp_nl_%d_nh_%d'%(args.nlayers, args.nhiddens)
    else:
        raise ValueError("what model ?")

    key, subkey = jax.random.split(key)
    loss = make_loss(hamiltonian_net)
    value_and_grad = jax.value_and_grad(loss)

    print("\n========== Prepare logs ==========")

    path = args.folder + "n_%d_dim_%d_beta_%g_lr_%g" % (args.n, args.dim, args.beta, args.lr) \
                       + "_" + modelname
    os.makedirs(path, exist_ok=True)
    print("Create directory: %s" % path)

    print("\n========== Load checkpoint==========")
    ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path or path) 
    if ckpt_filename is not None:
        print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
        ckpt = checkpoint.load_data(ckpt_filename)
        params = ckpt["params"]
    else:
        print("No checkpoint file found. Start from scratch.")

    print("\n========== Start training ==========")

    start = time.time()
    params = train(key, value_and_grad, args.epochs, args.batchsize, params, X1, args.lr, path, args.beta)
    end = time.time()
    running_time = end - start
    print('training time: %.5f sec' %running_time)
