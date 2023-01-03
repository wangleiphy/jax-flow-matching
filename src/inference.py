import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from data import sample_target
from net import make_vec_field_net, make_backflow, make_transformer
from energy import make_energy, make_free_energy
from train import train
from scale import make_scale
from pt import make_point_transformation
from flow import make_symplectic_flow
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
    group.add_argument('--emlp', action='store_true', help='emlp')

    group = parser.add_argument_group('physics parameters')
    group.add_argument('--n', type=int, default=6, help='The number of particles')
    group.add_argument('--dim', type=int, default=2, help='The dimensions of the system')
    group.add_argument('--beta', type=float, default=10.0, help='inverse temperature')

    args = parser.parse_args()

    energy_fn, potential_fn = make_energy(args.n, args.dim)

    print("\n========== Build networks ==========")
    key, subkey = jax.random.split(key)
    if args.backflow:
        print ('construct backflow network')
        v_params, vec_field_net = make_backflow(subkey, args.n, args.dim, [args.nhiddens]*args.nlayers)
        modelname = 'backflow_nl_%d_nh_%d'%(args.nlayers, args.nhiddens)
    elif args.transformer:
        print ('construct transformer network')
        v_params, vec_field_net = make_transformer(subkey, args.n, args.dim, args.nheads, args.nlayers, args.keysize)
        modelname = 'transformer_nl_%d_nh_%d_nk_%d'%(args.nlayers, args.nheads, args.keysize)
    elif args.mlp:
        print ('construct mlp network')
        v_params, vec_field_net = make_vec_field_net(subkey, args.n, args.dim, ch=args.nhiddens, num_layers=args.nlayers, symmetry=False)
        modelname = 'mlp_nl_%d_nh_%d'%(args.nlayers, args.nhiddens)
    elif args.emlp:
        print ('construct emlp network')
        v_params, vec_field_net = make_vec_field_net(subkey, args.n, args.dim, ch=args.nhiddens, num_layers=args.nlayers, symmetry=True)
        modelname = 'emlp'
    else:
        raise ValueError("what model ?")

    key, subkey = jax.random.split(key)
    s_params, scale_net = make_scale(subkey, 2*args.n*args.dim)
    pt = make_point_transformation(vec_field_net)
    sample_fn, _ = make_symplectic_flow(scale_net, pt, 2*args.n*args.dim, args.beta)
    free_energy_fn = make_free_energy(energy_fn, sample_fn, args.n, args.dim, args.beta)

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
    omega = jnp.exp(-params[0]['scale']['logscale'])
    omega = jnp.sort(omega)
    print ('omega:', omega)

    log_filename = os.path.join(folder, "epoch_%06d.txt" %(epoch_finished))
    f = open(log_filename, "w", buffering=1, newline="\n")
    f.write(("%6d" + "    %.6f"*4 + "\n") % (epoch_finished, fe, fe_err, vfe, vfe_err) )
    f.close()
    
    ckpt["x"] = x 
    ckpt["omega"] = omega
    checkpoint.save_data(ckpt, ckpt_filename)

    import matplotlib.pyplot as plt 
    plot_range = [(-2, 2), (-2, 2)]
    n_bins = 101
    fig = plt.figure(figsize=(18, 12))
    
    #sample target 
    key, subkey = jax.random.split(key)
    p = jax.random.normal(subkey, (args.batchsize, args.n*args.dim))
    p = p / jnp.sqrt(args.beta)
    key, subkey = jax.random.split(key)
    logp_fn = lambda q: -args.beta * potential_fn(q)
    q = sample_target(subkey, args.batchsize, args.n, args.dim, logp_fn)

    p = p.reshape(-1, args.dim)
    q = q.reshape(-1, args.dim)
    plt.subplot(2, 2, 1)
    plt.hist2d(p[:, 0], p[:, 1], bins=n_bins, range=plot_range, density=True, cmap="inferno")
    plt.subplot(2, 2, 2)
    plt.hist2d(q[:, 0], q[:, 1], bins=n_bins, range=plot_range, density=True, cmap="inferno")

    p, q = jnp.split(x, 2, axis=1)
    p = p.reshape(-1, args.dim)
    q = q.reshape(-1, args.dim)
    plt.subplot(2, 2, 3)
    plt.hist2d(p[:, 0], p[:, 1], bins=n_bins, range=plot_range, density=True, cmap="inferno")
    plt.subplot(2, 2, 4)
    plt.hist2d(q[:, 0], q[:, 1], bins=n_bins, range=plot_range, density=True, cmap="inferno")

    fig_filename = os.path.join(folder, "epoch_%06d.png" %(epoch_finished))
    plt.savefig(fig_filename)
