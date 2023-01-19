from config import * 

from nct import make_canonical_transformation
from net import make_hamiltonian_net

def test_reversibility():

    n = 6
    dim = 2

    init_rng, rng = jax.random.split(jax.random.PRNGKey(42))
    params, hamiltonian_net = make_hamiltonian_net(init_rng, n, dim)
    
    ct = make_canonical_transformation(hamiltonian_net)

    x0 = jax.random.normal(rng, (2*n*dim,))
    x1 = ct(params, x0, 1)

    assert x1.shape == x0.shape
    assert jnp.allclose(ct(params, x1, -1), x0)

def test_symplecity():

    n = 3
    dim = 2

    init_rng, rng = jax.random.split(jax.random.PRNGKey(42))
    params, hamiltonian_net = make_hamiltonian_net(init_rng, n, dim)

    ct = make_canonical_transformation(hamiltonian_net)
    
    x0 = jax.random.normal(rng, (2*n*dim,))
    M = jax.jacrev(ct, argnums=1)(params, x0, 1)
    
    assert jnp.allclose(jnp.linalg.det(M), 1.0) 
    
    zero = np.zeros((n*dim, n*dim))
    eye = np.eye(n*dim)
    J = np.bmat([[zero, eye], 
                 [-eye, zero]])
    
    assert jnp.allclose(M@J@M.T, J)

test_symplecity()
