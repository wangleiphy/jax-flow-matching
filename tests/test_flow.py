from config import *
from transformer import make_transformer 
from flow import make_flow
from loss import make_mle_loss

def test_logp():

    n = 4
    dim = 2
    L = 1.234
    nheads = 4
    nlayers = 2
    keysize = 4
    batchsize = 10

    key = jax.random.PRNGKey(42)

    params, network, div_fn = make_transformer(key, n, dim, nheads, nlayers, keysize, L)
    sampler, sampler_with_logp, logp_fn = make_flow(network, div_fn, n*dim, L)
    
    key, subkey = jax.random.split(key)

    x = sampler(subkey, params, batchsize)
    assert (x.shape == (batchsize, 5, n*dim))

    x, logp = sampler_with_logp(subkey, params, batchsize)
    assert (x.shape == (batchsize, n*dim))
    assert (logp.shape == (batchsize, ))
    subkeys = jax.random.split(key, batchsize)

    logp_inference = logp_fn(params, x, subkeys)
    assert (logp_inference.shape == (batchsize, ))
    print (logp)
    print (logp_inference)

    #assert jnp.allclose(logp, logp_inference)
    
    loss_fn = make_mle_loss(logp_fn)
    g = jax.grad(loss_fn)(params, x, subkeys)

test_logp()
