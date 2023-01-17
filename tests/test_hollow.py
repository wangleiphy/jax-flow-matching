from config import * 

from hollow import make_hollow_net, make_divergence_fn
from utils import divergence_fwd

def test_symmetry():
    n = 8
    dim = 3
    L = 1.234
    hidden_sizes = [16, 16]

    key = jax.random.PRNGKey(42)

    network = make_hollow_net(hidden_sizes, L)
    params = network.init(key, jnp.zeros((n, dim)))

    x = jax.random.uniform(key, (n, dim), minval=0, maxval=L)
    v = network.apply(params, x)

    P = np.random.permutation(n)
    Pv = network.apply(params, x[P, :])

    assert jnp.allclose(Pv, v[P, :])

    Tv = network.apply(params, x+L)
    assert jnp.allclose(Tv, v)

def test_div():
    n = 3
    dim = 2
    L = 1.234
    hidden_sizes = [16, 16]

    key = jax.random.PRNGKey(42)

    network = make_hollow_net(hidden_sizes, L)
    params = network.init(key, jnp.zeros((n, dim)))

    x = jax.random.uniform(key, (n, dim), minval=0, maxval=L)
    
    f = lambda x: network.apply(params, x.reshape(n, dim)).reshape(-1)
    div = divergence_fwd(f)(x.reshape(-1))
    
    div_fn = make_divergence_fn(network)
    div2 = div_fn(params, x)
    
    print (div, div2)
    print (jax.jacfwd(f)(x.reshape(-1)))
    assert (jnp.allclose(div, div2))

test_div()
