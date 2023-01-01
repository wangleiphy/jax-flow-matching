import jax 
import jax.numpy as jnp
import haiku as hk

from typing import Optional

class Scale(hk.Module):

    def __init__(self, 
                 name: Optional[str] = None
                ):
        super().__init__(name=name)
     
    def __call__(self, x, sign): 
        assert jnp.abs(sign) == 1.0
        p, q = jnp.split(x, 2)

        logscale = hk.get_parameter("logscale", [x.shape[0]//2, ], init=hk.initializers.TruncatedNormal(stddev=0.01), dtype=x.dtype)
        q = q*jnp.exp(sign*logscale)

        return jnp.concatenate([p, q]), jnp.sum(sign*logscale)
