import jax
import jax.numpy as jnp
from functools import partial

def make_loss(vec_field_net, L):

    @partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=0)
    def _matching(params, x0, x1, t):
        def circular_interpolation(x0, x1, t, L):
            if True:
                diff = x1-x0 
                return jnp.where(jnp.abs(diff)<L/2, 
                                 x0 + t*diff, 
                                 jnp.where(diff>0, x0+t*(diff-L), x0+t*(diff+L))
                                )
            else: 
                # wrapped cauchy p. 473. http://www.nrbook.com/devroye/Devroye_files/chapter_nine.pdf
                eps = 1e-3
                rho0, rho1 = eps, 1-eps
                rho = rho0 * (1-t) + rho1 * t
                mu = x1 * t 
                s = (1.0 + rho**2) / (2.0 * rho)
                u = 2*x0 - 1 # U(-1, 1)
                z = jnp.cos(jnp.pi*u)
                w = (1.0 + s * z) / (s + z)
                w = jnp.clip(w, a_min=-1+eps, a_max=1-eps) # clip to avoid numerical error 
                x = jnp.sign(u) * jnp.arccos(w) # (-pi, pi)
                return x*L/(2*jnp.pi) + mu

        x = circular_interpolation(x0, x1, t, L)
        v = jax.vmap(jax.grad(circular_interpolation, argnums=2), (0, 0, None, None), 0)(x0, x1, t, L)
        return jnp.sum((v - vec_field_net(params, x, t))**2)

    def loss(params, x0, x1, t):
        m = _matching(params, x0, x1, t)
        return jnp.mean(m)

    return loss

def make_mle_loss(logp_fn):
    def loss_fn(params, x, key):
        logp = logp_fn(params, x, key)
        return -jnp.mean(logp)
    return loss_fn
