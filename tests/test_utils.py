from config import * 

from utils import divergence_fwd, divergence_fori, divergence_scan, divergence_hutchinson

def test_div():

    n = 100
    key = jax.random.PRNGKey(42)
    w = jax.random.normal(key, (n, n))

    def f(x):
        y = w@x
        return jnp.sin(y) + y**2  +x 
    
    x = jax.random.normal(key, (n,))

    div1 = divergence_fwd(f)(x)

    div_scan = divergence_scan(f)(x)
    div_fori = divergence_fori(f)(x)

    assert jnp.allclose(div1, div_scan)
    assert jnp.allclose(div1, div_fori)
    
    batchsize = 4096
    keys = jax.random.split(key, batchsize)
    div2 = jax.vmap(divergence_hutchinson(f), (0, None))(keys, x)

    print (div1)
    print (jnp.mean(div2), '+/-', jnp.std(div2)/jnp.sqrt(batchsize))

    assert jnp.allclose(div1, jnp.mean(div2), rtol=1e-1)

test_div()
