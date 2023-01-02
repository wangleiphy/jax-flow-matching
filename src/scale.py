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
        p, q = jnp.split(x, 2)

        logscale = hk.get_parameter("logscale", [x.shape[0]//2, ], init=hk.initializers.TruncatedNormal(stddev=0.01), dtype=x.dtype)
        q = q*jnp.exp(sign*logscale)

        return jnp.concatenate([p, q]), jnp.sum(sign*logscale)

def make_scale(key, dim):

    def forward_fn(x, sign):
        net = Scale() 
        return net(x, sign)

    x = jax.random.normal(key, (dim,))    
    network = hk.transform(forward_fn)
    network = hk.without_apply_rng(network)
    params = network.init(key, x, 1)
    return params, network.apply
