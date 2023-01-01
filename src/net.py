import jax
import jax.numpy as jnp
from jax.example_libraries.stax import serial, Dense, Relu
from jax.nn.initializers import zeros
from jax import random
import haiku as hk
import emlp.nn.haiku as ehk
from emlp.reps import V
from emlp.groups import SO, Z
from emlp.reps import Rep
from emlp.nn import uniform_rep
from backflow import Backflow
from transformer import Transformer
import logging

def Sequential(*args):
    """ Wrapped to mimic pytorch syntax"""
    return lambda x: hk.Sequential(args)(x)

def EMLP_with_t(n, spatial_dim, ch=384, num_layers=3):
    """ SO(n) Equivariant MultiLayer Perceptron with time.
        If the input ch argument is an int, uses the hands off uniform_rep heuristic.
        If the ch argument is a representation, uses this representation for the hidden layers.
        Individual layer representations can be set explicitly by using a list of ints or a list of
        representations, rather than use the same for each hidden layer.
        Args:
            n (int): the number of particles
            spatial_dim (int): the number of spatial dimensions
            ch (int or list[int] or Rep or list[Rep]): number of channels in the hidden layers
            num_layers (int): number of hidden layers
        Returns:
            Module: the EMLP haiku module."""
    logging.info("Initing EMLP with time (Haiku)")

    group = SO(spatial_dim)
    rep_in = n*V(group)
    rep_out = n*V(group)

    # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
    if isinstance(ch,int): middle_layers = num_layers*[uniform_rep(ch,group)]
    elif isinstance(ch,Rep): middle_layers = num_layers*[ch(group)]
    else: middle_layers = [(c(group) if isinstance(c,Rep) else uniform_rep(c,group)) for c in ch]
    # assert all((not rep.G is None) for rep in middle_layers[0].reps)
    reps = [rep_in]+middle_layers
    # logging.info(f"Reps: {reps}")
    network = Sequential(
        ehk.Linear(rep_in+V(Z(1)), rep_in),
        *[ehk.EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
        ehk.Linear(reps[-1],rep_out)
    )
    return network

def MLP_with_t(n, spatial_dim, ch=384, num_layers=3):
    
    init = hk.initializers.Constant(0)
    middle_layers = num_layers*[ch]

    network = Sequential(
        lambda x: hk.nets.MLP(middle_layers)(x),
        lambda x: hk.Linear(n*spatial_dim, init, init)(x)
    )

    return network

def make_vec_field_net(rng, n, spatial_dim, ch=512, num_layers=2, symmetry=False):

    if symmetry:
        model = EMLP_with_t(n, spatial_dim, ch, num_layers)
    else:
        model = MLP_with_t(n, spatial_dim, ch, num_layers)

    def vec_field_net(x, t):
        input = jnp.concatenate((x,t.reshape(1)))
        return model(input)

    #return vec_field_net

    net = hk.without_apply_rng(hk.transform(vec_field_net))

    params = net.init(rng, jnp.ones((n*spatial_dim,)), jnp.ones((1,)))
    net_apply = net.apply

    return params, net_apply

def make_backflow(key, n, dim, sizes):
    x = jax.random.normal(key, (n, dim))
    t = jax.random.uniform(key)

    def forward_fn(x, t):
        net = Backflow(sizes)
        return net(x.reshape(n, dim), t).reshape(n*dim)
    network = hk.without_apply_rng(hk.transform(forward_fn))
    params = network.init(key, x, t)
    return params, network.apply 

def make_transformer(key, n, dim, num_heads, num_layers, key_sizes):
    x = jax.random.normal(key, (n, dim))
    t = jax.random.uniform(key)

    def forward_fn(x, t):
        net = Transformer(num_heads, num_layers, key_sizes)
        return net(x.reshape(n, dim), t).reshape(n*dim)
    network = hk.without_apply_rng(hk.transform(forward_fn))
    params = network.init(key, x, t)
    return params, network.apply 

if __name__ == '__main__':
    from jax.config import config
    config.update("jax_enable_x64", True)

    n = 6
    spatial_dim = 2

    params, vec_field_net = make_vec_field_net(random.PRNGKey(42), n, spatial_dim, symmetry=False)

    x = random.normal(random.PRNGKey(41), (n*spatial_dim,))
    t = random.normal(random.PRNGKey(40), (1,))

    import time
    start = time.time()
    v = vec_field_net(params,  x, t)
    end = time.time()
    print(end - start)
    print(v)
    print(vec_field_net(jax.tree_util.tree_map(lambda x: x -0.01, params),  x, t))
