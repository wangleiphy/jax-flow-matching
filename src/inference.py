import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from transformer import make_transformer
#from energy import make_energy, make_free_energy
from flow import make_flow 
import checkpoint
import utils 
import matplotlib.pyplot as plt 

import os
import time

if __name__ == '__main__':
    key = jax.random.PRNGKey(42)

    import argparse
    parser = argparse.ArgumentParser(description='')

    group = parser.add_argument_group('learning parameters')
    group.add_argument('--batchsize', type=int, default=4096, help='')

    group = parser.add_argument_group('filesystem')
    group.add_argument("--restore_path", default=None, help="checkpoint path or file")

    group = parser.add_argument_group('network parameters')
    group.add_argument('--nlayers', type=int, default=4, help='The number of layers')
    group.add_argument('--nheads', type=int, default=8, help='')
    group.add_argument('--keysize', type=int, default=16, help='')

    group = parser.add_argument_group('physics parameters')
    group.add_argument('--n', type=int, default=6, help='The number of particles')
    group.add_argument('--dim', type=int, default=2, help='The dimensions of the system')
    group.add_argument('--beta', type=float, default=10.0, help='inverse temperature')

    args = parser.parse_args()

    key = jax.random.PRNGKey(42)
    n = 32 
    dim = 3
    L = 12.225024745980599

    print("\n========== Build networks ==========")
    params, vec_field_net = make_transformer(key, n, dim, args.nheads, args.nlayers, args.keysize, L)
    modelname = "transformer_nl_%d_nh_%d_nk_%d" % (args.nlayers, args.nheads, args.keysize)

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
    print (x.shape, n, dim)
    rdf = utils.get_gr(x.reshape(-1, n, dim), L)
    plt.plot(rdf[0], rdf[1], linestyle='-', c='red', label='model')
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
