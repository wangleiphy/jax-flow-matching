from config import * 
from scale import Scale 

def make_scale(key, dim):

    def forward_fn(x, sign):
        net = Scale() 
        return net(x, sign)

    x = jax.random.normal(key, (dim, ))    
    network = hk.transform(forward_fn)
    network = hk.without_apply_rng(network)
    params = network.init(key, x, 1)
    return params, network.apply

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

