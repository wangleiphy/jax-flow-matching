from config import *
from transformer import make_transformer 
from flow import make_flow

def test_logp():

    n = 4
    dim = 2
    L = 1.234
    nheads = 4
    nlayers = 2
    keysize = 4
    batchsize = 10

    key = jax.random.PRNGKey(42)

    subkey0, subkey1, key = jax.random.split(key, 3)
    X0 = jax.random.normal(subkey0, (batchsize, n*dim))
    X1 = jax.random.normal(subkey1, (batchsize, n*dim))

    params, network, div_fn = make_transformer(key, n, dim, nheads, nlayers, keysize, L)
    sampler, sampler_with_logp = make_flow(network, div_fn, X0, X1)
    
    key, subkey = jax.random.split(key)

    x = sampler(subkey, params, batchsize, True)
    assert (x.shape == (batchsize, 5, n*dim))

    key, subkey = jax.random.split(key)
    x0, x1, logp = sampler_with_logp(subkey, params, batchsize, True)
    assert (x0.shape == (batchsize, n*dim))
    assert (x1.shape == (batchsize, n*dim))
    assert (logp.shape == (batchsize, ))
    print (logp)

    key, subkey = jax.random.split(key)
    x1, x0, logp = sampler_with_logp(subkey, params, batchsize, True)
    print (logp)

test_logp()
