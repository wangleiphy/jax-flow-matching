import jax
import jax.numpy as jnp
from functools import partial

from pt import phasespace_v

def make_loss(vec_field_net, beta):

    @partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=0)
    def _matching(params, x0, x1, t):
        x0 = x0 / jnp.sqrt(beta)
        x = t*x1 + (1 - t)*x0
        v = phasespace_v(params, vec_field_net, x, t)

        return jnp.sum(((x1 - x0) - v)**2)

    def loss_fn(params, x0, x1, t):
        m = _matching(params, x0, x1, t)
        return jnp.mean(m)

    return loss_fn
