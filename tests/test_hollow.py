from config import * 

from hollow import make_hollow_net
from utils import divergence_fwd

def test_symmetry():
    n = 8
    dim = 3
    L = 1.234
    nheads = 8
    keysize = 16 
    h1size = 32
    h2size = 32
    nlayers = 2 

    key = jax.random.PRNGKey(42)

    params, network, _ = make_hollow_net(key, n, dim, L, nheads, keysize, h1size, h2size, nlayers)

    x = jax.random.uniform(key, (n, dim), minval=0, maxval=L)
    t = jax.random.uniform(key)

    v = network(params, x, t).reshape(n, dim)

    P = np.random.permutation(n)
    Pv = network(params, x[P, :], t).reshape(n, dim)

    assert jnp.allclose(Pv, v[P, :])

    Tv = network(params, x+L, t).reshape(n, dim)
    assert jnp.allclose(Tv, v)

def test_div():
    n = 3
    dim = 2
    L = 1.234
    nheads = 8
    keysize = 16 
    h1size = 32
    h2size = 32
    nlayers = 2 

    key = jax.random.PRNGKey(42)

    params, network, div_fn = make_hollow_net(key, n, dim, L, nheads, keysize, h1size, h2size, nlayers)

    x = jax.random.uniform(key, (n, dim), minval=0, maxval=L)
    t = jax.random.uniform(key)
    
    f = lambda x: network(params, x.reshape(n, dim), t).reshape(-1)
    div = divergence_fwd(f)(x.reshape(-1))
    
    div2 = div_fn(params, x, t)
    
    print (div, div2)
    print (jax.jacfwd(f)(x.reshape(-1)))
    assert (jnp.allclose(div, div2))

test_div()
