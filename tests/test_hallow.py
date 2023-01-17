from config import * 

from hallow import make_hallow_net, make_divergence_fn
from utils import divergence_fwd

def test_symmetry():
    n = 8
    dim = 3
    hidden_sizes = [16, 16]

    key = jax.random.PRNGKey(42)

    network = make_hallow_net(hidden_sizes)
    params = network.init(key, jnp.zeros((n, dim)))

    x = jax.random.normal(key, (n, dim))
    v = network.apply(params, x)

    P = np.random.permutation(n)
    Pv = network.apply(params, x[P, :])

    assert jnp.allclose(Pv, v[P, :])

def test_div():
    n = 4
    dim = 3
    hidden_sizes = [16, 16]

    key = jax.random.PRNGKey(42)

    network = make_hallow_net(hidden_sizes)
    params = network.init(key, jnp.zeros((n, dim)))

    x = jax.random.normal(key, (n, dim))
    
    f = lambda x: network.apply(params, x.reshape(n, dim)).reshape(-1)
    div = divergence_fwd(f)(x.reshape(-1))
    
    div_fn = make_divergence_fn(network, n, dim)
    div2 = div_fn(params, x)
    
    print (div, div2)
    assert (jnp.allclose(div, div2))

test_div()
