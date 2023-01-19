from config import * 

from nct import make_canonical_transformation

from jax.example_libraries.stax import serial, Dense, Sigmoid 

def make_potential_net(rng, n, dim):
    net_init, net_apply = serial(Dense(512), Sigmoid, Dense(512), Sigmoid, Dense(1))
    in_shape = (-1, n*dim+1)
    _, net_params = net_init(rng, in_shape)

    def net_with_t(params, x, t):
        return net_apply(params, jnp.concatenate((x,t.reshape(1))))[0]
    
    return net_params, net_with_t

def test_reversibility():

    n = 6
    dim = 2

    init_rng, rng = jax.random.split(jax.random.PRNGKey(42))
    params, potential_net = make_potential_net(init_rng, n, dim)
    
    ct = make_canonical_transformation(potential_net)

    x0 = jax.random.normal(rng, (2*n*dim,))
    x1 = ct(params, x0, 1)

    assert x1.shape == x0.shape
    assert jnp.allclose(ct(params, x1, -1), x0)

def test_symplecity():

    n = 3
    dim = 2

    init_rng, rng = jax.random.split(jax.random.PRNGKey(42))
    params, potential_net = make_potential_net(init_rng, n, dim)

    ct = make_canonical_transformation(potential_net)
    
    x0 = jax.random.normal(rng, (2*n*dim,))
    M = jax.jacrev(ct, argnums=1)(params, x0, 1)
    
    assert jnp.allclose(jnp.linalg.det(M), 1.0) 
    
    zero = np.zeros((n*dim, n*dim))
    eye = np.eye(n*dim)
    J = np.bmat([[zero, eye], 
                 [-eye, zero]])
    
    assert jnp.allclose(M@J@M.T, J)

test_reversibility()
