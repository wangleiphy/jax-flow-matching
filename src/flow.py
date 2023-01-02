import jax 
import jax.numpy as jnp
from jax.scipy.stats import norm
from functools import partial

def make_symplectic_flow(scale, pt, dim, beta):

    batched_scale = jax.vmap(scale, (None, 0, None), (0, 0))
    batched_pt = jax.vmap(pt, (None, 0, None), 0)

    def sample(scale_params, pt_params, batchsize, key):
        x = jax.random.normal(key, (batchsize, dim))
        logp = norm.logpdf(x).sum(-1)
        x = x / jnp.sqrt(beta)
        x, logjac = batched_scale(scale_params, x, 1)
        x = batched_pt(pt_params, x, 1)
        return x, logp - logjac
    
    @partial(jax.vmap, in_axes=(None, None, 0), out_axes=(0, 0))
    def logp(scale_params, pt_params, x):
        x = pt(pt_params, x, -1)
        z, logjac = scale(scale_params, x, -1)
        z = z * jnp.sqrt(beta)
        return norm.logpdf(z).sum() + logjac, x # x is data at the prior (not base)

    return sample, logp
