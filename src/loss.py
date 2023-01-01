import jax.numpy as jnp
from jax import vmap
from functools import partial

def make_loss(vec_field_net):

    @partial(vmap, in_axes=(None, 0, 0, 0), out_axes=0)
    def _matching(params, x0, x1, t):
        x = t*x1 + (1 - t)*x0
        return jnp.sum(((x1 - x0) - vec_field_net(params, x, t))**2)

    def loss(params, x0, x1, t):
        m = _matching(params, x0, x1, t)
        m_mean = jnp.mean(m)
        m_std = jnp.std(m) / jnp.sqrt(m.shape[0])
        return m_mean, m_std

    return loss

if __name__ == '__main__':
    from jax import random
    from net import make_vec_field_net

    n = 2
    dim = 2
    sample_size = 10

    params, vec_field_net = make_vec_field_net(random.PRNGKey(42), n, dim, symmetry=False)

    loss = make_loss(vec_field_net)

    print(loss(params, random.normal(random.PRNGKey(41), (sample_size, n*dim)), jnp.ones((sample_size, n*dim)), jnp.zeros((sample_size,))))
