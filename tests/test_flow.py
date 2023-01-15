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

    params, network = make_transformer(key, n, dim, nheads, nlayers, keysize, L)
    sampler, sampler_with_logp = make_flow(network, n*dim, L)

    key, subkey = jax.random.split(key)
    x, logp = sampler_with_logp(subkey, params, batchsize)
    assert (x.shape == (batchsize, n*dim))
    assert (logp.shape == (batchsize, ))

test_logp()
