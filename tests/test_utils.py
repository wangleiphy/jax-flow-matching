from config import * 

from utils import divergence_fwd, divergence_hutchinson

def test_div():

    def f(x):
        return jnp.sin(x) + x**2 
    
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (10,))

    div1 = divergence_fwd(f)(x)
    
    batchsize = 8192
    keys = jax.random.split(key, batchsize)
    div2 = jax.vmap(divergence_hutchinson(f), (0, None))(keys, x)

    print (div1)
    print (jnp.mean(div2))

    assert jnp.allclose(div1, jnp.mean(div2), rtol=1e-2)

test_div()
