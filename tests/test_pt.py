from config import * 

from pt import make_point_transformation

from jax.example_libraries.stax import serial, Dense, Sigmoid 

def make_vec_field_net(rng, n, dim):
    net_init, net_apply = serial(Dense(512), Sigmoid, Dense(512), Sigmoid, Dense(n*dim))
    in_shape = (-1, n*dim)
    _, net_params = net_init(rng, in_shape)

    def net(params, x):
        return net_apply(params, x)
    
    return net_params, net

def test_reversibility():

    n = 6
    dim = 2

    init_rng, rng = jax.random.split(jax.random.PRNGKey(42))
    v_params, vec_field_net = make_vec_field_net(init_rng, n, dim)
    s_params = jax.random.normal(init_rng, (n*dim, ))* 0.01
    
    params = s_params, v_params
    point_transformation = make_point_transformation(vec_field_net)

    x0 = jax.random.normal(rng, (2*n*dim,))
    x1 = point_transformation(params, x0, 1)

    assert x1.shape == x0.shape
    assert jnp.allclose(point_transformation(params, x1, -1), x0)

def test_symplecity():

    n = 3
    dim = 2

    init_rng, rng = jax.random.split(jax.random.PRNGKey(42))
    v_params, vec_field_net = make_vec_field_net(init_rng, n, dim)
    s_params = jax.random.normal(init_rng, (n*dim, ))

    params = s_params, v_params
    point_transformation = make_point_transformation(vec_field_net)
    
    x0 = jax.random.normal(rng, (2*n*dim,))
    M = jax.jacrev(point_transformation, argnums=1)(params, x0, 1)

    t = np.full((n*dim, n*dim), True)
    f = np.full((n*dim, n*dim), False)
    J = np.bmat([[t, t], 
                 [f, t]])
    assert jnp.alltrue((M!=0)==J)
    assert jnp.allclose(jnp.linalg.slogdet(M)[1], jnp.sum(s_params))
    
    zero = np.zeros((n*dim, n*dim))
    eye = np.eye(n*dim)
    J = np.bmat([[zero, eye], 
                 [-eye, zero]])

    s = jnp.diag(jnp.concatenate([jnp.exp(s_params), jnp.ones_like(s_params)]))
    assert jnp.allclose(M@J@M.T, s@J@s, atol=1e-2)

test_symplecity()
