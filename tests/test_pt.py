from config import * 

from pt import PointTransformation

from jax.example_libraries.stax import serial, Dense, Relu


def make_vec_field_net(rng, n, dim):
    net_init, net_apply = serial(Dense(512), Relu, Dense(512), Relu, Dense(n*dim))
    in_shape = (-1, n*dim+1)
    _, net_params = net_init(rng, in_shape)

    def net_with_t(params, x, t):
        return net_apply(params, jnp.concatenate((x,t.reshape(1))))
    
    return net_params, net_with_t

def test_reversibility():

    n = 6
    dim = 2

    init_rng, rng = jax.random.split(jax.random.PRNGKey(42))
    params, vec_field_net = make_vec_field_net(init_rng, n, dim)

    forward, reverse = PointTransformation(vec_field_net)

    x0 = jax.random.normal(rng, (2*n*dim,))
    x1 = forward(params, x0)

    assert x1.shape == x0.shape
    assert jnp.allclose(reverse(params, x1), x0)

def test_simplecity():

    n = 2
    dim = 2

    init_rng, rng = jax.random.split(jax.random.PRNGKey(42))
    params, vec_field_net = make_vec_field_net(init_rng, n, dim)

    forward, reverse = PointTransformation(vec_field_net)
    
    x0 = jax.random.normal(rng, (2*n*dim,))
    M = jax.jacrev(forward, argnums=1)(params, x0)
    print (M!=0)

    assert jnp.allclose(jnp.linalg.det(M), 1.0)
    
    zero = np.zeros((n*dim, n*dim))
    eye = np.eye(n*dim)
    J = np.bmat([[zero, eye], 
                 [-eye, zero]])
    assert jnp.allclose(M@J@M.T, J, atol=1e-6)
