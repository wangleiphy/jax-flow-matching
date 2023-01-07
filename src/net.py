import jax
import jax.numpy as jnp
from jax.example_libraries.stax import serial, Dense, Relu
from jax.nn.initializers import zeros
from jax import random
import haiku as hk
import emlp.nn.haiku as ehk
from backflow import Backflow
from transformer import Transformer
import logging

def Sequential(*args):
    """ Wrapped to mimic pytorch syntax"""
    return lambda x: hk.Sequential(args)(x)

def MLP(n, spatial_dim, ch=384, num_layers=3):
    
    init = hk.initializers.Constant(0)
    middle_layers = num_layers*[ch]

    network = Sequential(
        lambda x: hk.nets.MLP(middle_layers)(x),
        lambda x: hk.Linear(n*spatial_dim, init, init)(x)
    )

    return network

def make_vec_field_net(rng, n, spatial_dim, ch=512, num_layers=2):

    model = MLP(n, spatial_dim, ch, num_layers)

    def vec_field_net(x):
        return model(x)

    net = hk.without_apply_rng(hk.transform(vec_field_net))

    params = net.init(rng, jnp.ones((n*spatial_dim,)))
    net_apply = net.apply

    return params, net_apply

def make_backflow(key, n, dim, sizes):
    x = jax.random.normal(key, (n, dim))

    def forward_fn(x):
        net = Backflow(sizes)
        return net(x.reshape(n, dim)).reshape(n*dim)
    network = hk.without_apply_rng(hk.transform(forward_fn))
    params = network.init(key, x)
    return params, network.apply 

def make_transformer(key, n, dim, num_heads, num_layers, key_sizes):
    x = jax.random.normal(key, (n, dim))
    t = jax.random.uniform(key)

    def forward_fn(x):
        net = Transformer(num_heads, num_layers, key_sizes)
        return net(x.reshape(n, dim)).reshape(n*dim)
    network = hk.without_apply_rng(hk.transform(forward_fn))
    params = network.init(key, x)
    return params, network.apply 
