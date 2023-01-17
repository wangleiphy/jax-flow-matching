from config import *
from transformer import make_transformer 
from flow import make_flow

def test_logp():

    n = 6
    dim = 2
    L = 1.234
    nheads = 8 
    nlayers = 4
    keysize = 16 
    batchsize = 10

    key = jax.random.PRNGKey(42)

    params, network, div_fn = make_transformer(key, n, dim, nheads, nlayers, keysize, L)
    sampler, sampler_with_logp = make_flow(network, div_fn, n*dim, L)
    
    key, subkey = jax.random.split(key)

    x = sampler(subkey, params, batchsize)
    assert (x.shape == (batchsize, 5, n*dim))

    x, logp = sampler_with_logp(subkey, params, batchsize)
    assert (x.shape == (batchsize, n*dim))
    assert (logp.shape == (batchsize, ))

test_logp()
