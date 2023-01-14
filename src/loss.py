import jax
import jax.numpy as jnp
from functools import partial

def make_loss(vec_field_net):

    @partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=0)
    def _matching(params, x0, x1, t):
        x = t*x1 + (1 - t)*x0
        return jnp.sum(((x1 - x0) - vec_field_net(params, x, t))**2)

    def loss(params, x0, x1, t):
        m = _matching(params, x0, x1, t)
        return jnp.mean(m)

    return loss
