import jax 
import jax.numpy as jnp
import numpy as np
from functools import partial

def make_symplectic_flow(pt, dim, beta):

    batched_pt = jax.vmap(pt, (None, 0, None), 0)
    
    logp_fn = lambda x: -0.5*(x**2 + jnp.log(2 * np.pi/beta))

    def sample(key, params, batchsize):
        x = jax.random.normal(key, (batchsize, dim))
        logp = logp_fn(x).sum(-1)
        x = x / jnp.sqrt(beta)
        x = batched_pt(params, x, 1)
        return x, logp - jnp.sum(params[0])
    
    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def logp(params, x):
        z = pt(params, x, -1)
        z = z * jnp.sqrt(beta)
        return logp_fn(z).sum() - jnp.sum(params[0])

    return sample, logp
