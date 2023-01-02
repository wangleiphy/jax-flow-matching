import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from data import sample_target
from net import make_vec_field_net, make_backflow, make_transformer
from loss import make_loss
from energy import make_energy, make_free_energy
from train import train

import time

if __name__ == '__main__':
    key = jax.random.PRNGKey(42)

    import argparse
    parser = argparse.ArgumentParser(description='')

    group = parser.add_argument_group('learning parameters')
    group.add_argument('-epoch', type=int, default=200, help='')
    group.add_argument('-batchsize', type=int, default=4096, help='')
    group.add_argument('-lr', type=float, default=1e-3, help='learning rate')

    group = parser.add_argument_group('datasets')
    group.add_argument('-datasize', type=int, default=102400, help='')
    group.add_argument('-name', type=str, default='mcmc', help='')

    group = parser.add_argument_group('network parameters')
    group.add_argument('-nhiddens', type=int, default=512, help='The channels in a middle layer')
    group.add_argument('-nlayers', type=int, default=4, help='The number of layers')
    group.add_argument('-nheads', type=int, default=8, help='')
    group.add_argument('-keysize', type=int, default=16, help='')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-backflow', action='store_true', help='Use backflow')
    group.add_argument('-transformer', action='store_true', help='Use transformer')
    group.add_argument('-mlp', action='store_true', help='mlp')
    group.add_argument('-emlp', action='store_true', help='emlp')

    group = parser.add_argument_group('physics parameters')
    group.add_argument('-n', type=int, default=6, help='The number of particles')
    group.add_argument('-dim', type=int, default=2, help='The dimensions of the system')
    group.add_argument('-beta', type=float, default=10.0, help='')

    args = parser.parse_args()

    '''generating datasets'''
    start = time.time()
    energy_fn, potential_fn = make_energy(args.n, args.dim)
    logp_fn = lambda q: -args.beta * potential_fn(q)
    X1 = sample_target(key, args.datasize, args.n, args.dim, logp_fn)
    end = time.time()
    running_time = end - start
    print('training set sampling time: %.5f sec' %running_time)

    '''building networks'''
    key, subkey = jax.random.split(key)
    if args.backflow:
        print ('construct backflow network')
        v_params, vec_field_net = make_backflow(subkey, args.n, args.dim, [args.channel]*args.numlayers)
    elif args.transformer:
        print ('construct transformer network')
        v_params, vec_field_net = make_transformer(subkey, args.n, args.dim, args.nheads, args.numlayers, args.keysize)
    elif args.mlp:
        print ('construct mlp network')
        v_params, vec_field_net = make_vec_field_net(subkey, args.n, args.dim, ch=args.channel, num_layers=args.numlayers, symmetry=False)
    elif args.emlp:
        print ('construct emlp network')
        v_params, vec_field_net = make_vec_field_net(subkey, args.n, args.dim, ch=args.channel, num_layers=args.numlayers, symmetry=True)
    else:
        raise ValueError("what model ?")

    '''initializing the sampler and logp calculator'''
    
    key, subkey = jax.random.split(key)
    s_params, scale = make_scale(subkey, 2*args.n*args.dim)
    pt = make_point_transformation(vec_field_net)
    sample_fn, logp_fn = make_symplectic_flow(scale, pt, 2*args.n*dim, args.beta)
    free_energy_fn = make_free_energy(energy_fn, sample_fn, logp_fun, args.n, args.dim, args.beta)

    '''initializing the loss function'''
    loss = make_loss(scale_net, vec_field_net)
    value_and_grad = jax.value_and_grad(loss, argnums=0, has_aux=True)
    
    start = time.time()
    params = (s_params, v_params)
    trained_params = train(key, args.epoch, params, X1, args.lr)
    end = time.time()
    running_time = end - start
    print('training time: %.5f sec' %running_time)

    start = time.time()
    key, subkey = jax.random.split(key)
    fe, fe_err, X_syn, f, f_err = free_energy(subkey, trained_params, args.batchsize)
    end = time.time()
    running_time = end - start
    print('free energy using trained model: %f ± %f' %(fe, fe_err))
    print('variational free energy using trained model: %f ± %f' %(f, f_err))
    print('importance sampling time: %.5f sec' %running_time)
