from config import * 
from pt import make_point_transformation
from flow import make_symplectic_flow

from scale import make_scale
from test_pt import make_vec_field_net

def test_logp():

    n = 6
    dim = 2
    batchsize = 10
    T = 1.0
    key = jax.random.PRNGKey(42)

    scale_params, scale = make_scale(key, 2*n*dim)
    pt_params, vec_field_net = make_vec_field_net(key, n, dim)
    
    params = scale_params, pt_params

    pt = make_point_transformation(vec_field_net)

    sample, logp_fn = make_symplectic_flow(scale, pt, 2*n*dim, T)
    
    key, subkey = jax.random.split(key)
    x, logp = sample(subkey, params, batchsize)
    assert (x.shape == (batchsize, 2*n*dim))
    assert (logp.shape == (batchsize, ))

    logp_inference, _ = logp_fn(params, x)
    
    assert jnp.allclose(logp, logp_inference) 
