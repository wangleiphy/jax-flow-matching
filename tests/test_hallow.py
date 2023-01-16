from config import * 
from hallow import make_hallow_net

from utils import divergence_fwd

def test_symmetry():
    n = 4 
    dim = 1
    hidden_sizes = [16, 16]

    key = jax.random.PRNGKey(42)

    network = make_hallow_net(hidden_sizes)
    params = network.init(key, jnp.zeros((n, dim)))

    x = jax.random.normal(key, (n, dim))
    v, div = network.apply(params, x)

    P = np.random.permutation(n)
    Pv, Pdiv = network.apply(params, x[P, :])

    print ('x', x)
    print ('Px', x[P,:]) 
    
    #print (v)
    #print (Pv)

    #print (div)
    #print (Pdiv)

    assert jnp.allclose(Pv, v[P, :])

def test_div():
    n = 10
    dim = 3
    hidden_sizes = [16, 16]

    key = jax.random.PRNGKey(42)

    network = make_hallow_net(hidden_sizes)
    params = network.init(key, jnp.zeros((n, dim)))

    x = jax.random.normal(key, (n, dim))
    
    f = lambda x: network.apply(params, x.reshape(n, dim))[0].reshape(-1)
    div = divergence_fwd(f)(x.reshape(-1))
    print (div)
    print (network.apply(params, x)[1])

test_symmetry()
