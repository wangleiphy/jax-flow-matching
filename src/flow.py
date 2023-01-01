import jax 
import jax.numpy as jnp
from jax.scipy.stats import norm
from functools import partial

def make_symplectic_flow(scale, pt, dim):

    batched_scale = jax.vmap(scale, (None, 0, None), (0, 0))
    batched_pt = jax.vmap(pt, (None, 0, None), 0)

    def sample(scale_params, pt_params, batchsize, key):
        x = jax.random.normal(key, (batchsize, dim))
        logp = norm.logpdf(x).sum(-1)
        x, logjac = batched_scale(scale_params, x, 1)
        x = batched_pt(pt_params, x, 1)
        return x, logp - logjac
    
    @partial(jax.vmap, in_axes=(None, None, 0), out_axes=0)
    def logp(scale_params, pt_params, x):
        x = pt(pt_params, x, -1)
        x, logjac = scale(scale_params, x, -1)
        return norm.logpdf(x).sum() + logjac

    return sample, logp
