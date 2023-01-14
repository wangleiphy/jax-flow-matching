from config import * 
from transformer import make_transformer
   
def test_symmetry():
    n = 32
    dim = 3
    nheads = 8 
    nlayers = 4
    keysize = 16 
    L = 1.234

    key = jax.random.PRNGKey(42)
    params, network = make_transformer(key, n, dim, nheads, nlayers, keysize, L)
   
    x = jax.random.uniform(key, (n, dim), minval=0, maxval=L)
    t = jax.random.uniform(key)

    v = network(params, x, t).reshape(n, dim)
    P = np.random.permutation(n)
    Pv = network(params, x[P, :], t).reshape(n, dim)
    
    assert jnp.allclose(Pv, v[P, :])

    Tv = network(params, x+L, t).reshape(n, dim)
    assert jnp.allclose(Tv, v)
