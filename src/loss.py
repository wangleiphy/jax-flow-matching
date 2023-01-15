import jax
import jax.numpy as jnp
from functools import partial

def make_loss(vec_field_net, L):

    @partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=0)
    def _matching(params, x0, x1, t):
        def circular_interpolation(x0, x1, t, L):
            diff = x1-x0 
            return jnp.where(jnp.abs(diff)<L/2, 
                             x0 + t*diff, 
                             jnp.where(diff>0, x0+t*(diff-L), x0+t*(diff+L))
                            )

        x = circular_interpolation(x0, x1, t, L)
        v = jax.vmap(jax.grad(circular_interpolation, argnums=2), (0, 0, None, None), 0)(x0, x1, t, L)
        return jnp.sum((v - vec_field_net(params, x, t))**2)

    def loss(params, x0, x1, t):
        m = _matching(params, x0, x1, t)
        return jnp.mean(m)

    return loss
