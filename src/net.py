import jax
import jax.numpy as jnp
import haiku as hk
from backflow import Backflow
from transformer import Transformer

def Sequential(*args):
    """ Wrapped to mimic pytorch syntax"""
    return lambda x: hk.Sequential(args)(x)

def MLP_with_t(n, spatial_dim, ch=384, num_layers=3):
    
    init = hk.initializers.Constant(0)
    middle_layers = num_layers*[ch]

    network = Sequential(
        lambda x: hk.nets.MLP(middle_layers)(x),
        lambda x: hk.Linear(n*spatial_dim, init, init)(x)
    )

    return network

def make_hamiltonian_net(rng, n, spatial_dim, ch=512, num_layers=2):

    model = MLP_with_t(n, spatial_dim, ch, num_layers)

    def hamiltonian_net(p, q, t):
        input = jnp.concatenate((p, q, t.reshape(1)))
        return model(input).sum()

    net = hk.without_apply_rng(hk.transform(hamiltonian_net))

    params = net.init(rng, jnp.ones((n*spatial_dim,)), jnp.ones((n*spatial_dim,)), jnp.ones((1,)))
    net_apply = net.apply

    return params, net_apply

def make_backflow(key, n, dim, sizes):
    x = jax.random.normal(key, (n, dim))
    t = jax.random.uniform(key)

    def forward_fn(x, t):
        net = Backflow(sizes)
        v = net(x.reshape(n, dim), t).reshape(n*dim)
        return jnp.sum(v*v)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    params = network.init(key, x, t)
    return params, network.apply 

def make_transformer(key, n, dim, num_heads, num_layers, key_sizes):
    x = jax.random.normal(key, (n, dim))
    t = jax.random.uniform(key)

    def forward_fn(x, t):
        net = Transformer(num_heads, num_layers, key_sizes)
        v = net(x.reshape(n, dim), t).reshape(n*dim)
        return jnp.sum(v*v)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    params = network.init(key, x, t)
    return params, network.apply 
