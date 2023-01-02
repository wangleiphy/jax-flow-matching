from config import * 
from scale import make_scale

def test_reversibility():
    
    key = jax.random.PRNGKey(42)
    dim = 6

    params, scale = make_scale(key, dim)

    x = jax.random.normal(key, (dim, ))    
    y, logjacdet_fwd = scale(params, x, 1)
    x_bck, logjacdet_bck = scale(params, y, -1)
    
    print (params)
    assert jnp.allclose(x, x_bck)
    assert jnp.allclose(logjacdet_fwd, -logjacdet_bck)

