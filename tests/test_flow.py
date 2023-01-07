from config import * 
from pt import make_point_transformation
from flow import make_symplectic_flow
from net import make_transformer

def test_logp():

    n = 6
    dim = 2
    batchsize = 10
    beta = 10.0
    key = jax.random.PRNGKey(42)

    s_params = jax.random.normal(key, (n*dim, ))* 0.01
    v_params, vec_field_net = make_transformer(key, n, dim, 8, 4, 16)
    
    params = s_params, v_params

    pt = make_point_transformation(vec_field_net)

    sample, logp_fn = make_symplectic_flow(pt, 2*n*dim, beta)
    
    key, subkey = jax.random.split(key)
    x, logp = sample(subkey, params, batchsize)
    assert (x.shape == (batchsize, 2*n*dim))
    assert (logp.shape == (batchsize, ))

    logp_inference = logp_fn(params, x)
    
    assert jnp.allclose(logp, logp_inference) 

test_logp()
