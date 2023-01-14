from config import *
from transformer import make_transformer 
from flow import NeuralODE

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
    _, _, batched_sampler, logp_fun = NeuralODE(network, n*dim)

    key, subkey = jax.random.split(key)
    x, logp = batched_sampler(subkey, params, batchsize)
    assert (x.shape == (batchsize, n*dim))
    assert (logp.shape == (batchsize, ))

    logp_inference = logp_fun(params, x)
    
    assert jnp.allclose(logp, logp_inference) 

test_logp()
