import jax 
import jax.numpy as jnp
import numpy as np
from functools import partial

def make_symplectic_flow(ct, dim):

    batched_ct = jax.vmap(ct, (None, 0, None), 0)
    
    logp_fn = lambda x: -0.5*(x**2 + jnp.log(2 * np.pi))

    def sample(key, params, batchsize):
        x = jax.random.normal(key, (batchsize, dim))
        logp = logp_fn(x).sum(-1)
        x = batched_ct(params, x, 1)
        return x, logp 
    
    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def logp(params, x):
        z = ct(params, x, -1)
        return logp_fn(z).sum() 

    return sample, logp
