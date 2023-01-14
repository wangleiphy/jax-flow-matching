import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from transformer import make_transformer
from energy import make_energy, make_free_energy
from flow import NeuralODE
import checkpoint

import os
import time

if __name__ == '__main__':
    key = jax.random.PRNGKey(42)

    import argparse
    parser = argparse.ArgumentParser(description='')

    group = parser.add_argument_group('learning parameters')
    group.add_argument('--epochs', type=int, default=500, help='')
    group.add_argument('--batchsize', type=int, default=4096, help='')
    group.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    group = parser.add_argument_group('filesystem')
    group.add_argument("--restore_path", default=None, help="checkpoint path or file")

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

    energy_fn, potential_fn = make_energy(args.n, args.dim)

    print("\n========== Build networks ==========")
    params, vec_field_net = make_transformer(init_rng, n, dim, args.nheads, args.nlayers, args.keysize, L)
    modelname = "transformer_nl_%d_nh_%d_nk_%d" % (args.nlayers, args.nheads, args.keysize)

    key, subkey = jax.random.split(key)
    -_, _, batched_sample_fun, _ = NeuralODE(vec_field_net, args.n*args.dim)
    free_energy_fn = make_free_energy(energy_fn, batched_sample_fn, args.n, args.dim, args.beta)

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
    fe, fe_err, x, vfe, vfe_err = free_energy_fn(subkey, params, args.batchsize)
    end = time.time()
    running_time = end - start
    print('free energy using trained model: %f ± %f' %(fe, fe_err))
    print('variational free energy using trained model: %f ± %f' %(vfe, vfe_err))
    print('importance sampling time: %.5f sec' %running_time)
