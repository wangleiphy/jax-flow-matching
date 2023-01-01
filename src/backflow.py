import jax
import numpy as np 
import jax.numpy as jnp 
import haiku as hk
from typing import Optional

class Backflow(hk.Module):

    def __init__(self, 
                 sizes, 
                 name: Optional[str] = None
                  ):
        super().__init__(name=name)
        self.xi = jax.vmap(PseudoPotential(sizes), (0, None))
        self.eta = jax.vmap(jax.vmap(PseudoPotential(sizes), (0, None)), (0, None))
     
    def __call__(self, x, t): 
        '''
        Eq.(4.2) of https://global-sci.org/intro/article_detail/jml/20371.html 
        '''
        n, dim = x.shape[0], x.shape[1]

        r = jnp.linalg.norm(x, axis=-1)
        res = self.xi(r,t)*x

        rij = (jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))
        r = jnp.linalg.norm(rij + jnp.eye(n)[..., None], axis=-1)*(1.0-jnp.eye(n))

        res += (self.eta(r, t)*rij).sum(axis=1)
        return res

class PseudoPotential(hk.Module):
    def __init__(self, sizes, name=None):
        super().__init__(name=name)
        self.sizes = sizes
    
    def __call__(self, r : float, t : float) -> float:
        mlp = hk.nets.MLP(self.sizes + [1], activation=jax.nn.softplus, w_init=hk.initializers.TruncatedNormal(0.01))
        return mlp(jnp.stack([r, t]))

if __name__=='__main__':
    import jax 
    from jax.config import config   
    config.update("jax_enable_x64", True)
    import numpy as np
    n = 6
    dim = 2
    sizes = [64, 64]
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (n, dim))
    t = jax.random.uniform(key)

    def forward_fn(x, t):
        net = Backflow(sizes)
        return net(x, t)
    network = hk.transform(forward_fn)
    params = network.init(key, x, t)
    
    v = network.apply(params, None, x, t)
    P = np.random.permutation(n)
    Pv = network.apply(params, None, x[P, :], t)

    assert jnp.allclose(Pv, v[P, :])
