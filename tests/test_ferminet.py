from config import * 
from ferminet import make_ferminet 
   
def test_symmetry():
    n = 32
    dim = 3
    depth = 3 
    h1size = 32
    h2size = 16
    L = 1.234

    key = jax.random.PRNGKey(42)
    params, network, _ = make_ferminet(key, n, dim, depth, h1size, h2size, L, lambda x: jnp.sum(x**2))
   
    x = jax.random.uniform(key, (n, dim), minval=0, maxval=L)
    t = jax.random.uniform(key)

    v = network(params, x, t).reshape(n, dim)
    P = np.random.permutation(n)
    Pv = network(params, x[P, :], t).reshape(n, dim)

    assert jnp.allclose(Pv, v[P, :])

    Tv = network(params, x+L, t).reshape(n, dim)
    assert jnp.allclose(Tv, v)

test_symmetry()
