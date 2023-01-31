import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk
from typing import Optional

from utils import softcore
from utils import divergence_hutchinson as div

class FermiNet(hk.Module):

    def __init__(self, 
                 depth :int,
                 h1_size:int, 
                 h2_size:int, 
                 Nf:int,
                 L:float,
                 energy_fn, 
                 init_stddev:float = 0.01,
                 name: Optional[str] = None
                 ):
        super().__init__(name=name)
        assert (depth >= 2)
        self.depth = depth
        self.Nf = Nf
        self.L = L
        self.init_stddev = init_stddev
        self.energy_fn = energy_fn
  
        self.fc1 = [hk.Linear(h1_size, w_init=hk.initializers.TruncatedNormal(self.init_stddev)) for d in range(depth)]
        self.fc2 = [hk.Linear(h2_size, w_init=hk.initializers.TruncatedNormal(self.init_stddev)) for d in range(depth-1)]
    
    def _h2(self, x):
        n, dim = x.shape[0], x.shape[1]
        rij = (jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))
        
        #|r| calculated with periodic consideration
        r = jnp.linalg.norm(jnp.sin(np.pi*rij/self.L)+jnp.eye(n)[..., None], axis=-1)*(1.0-jnp.eye(n))
        
        f = [r[..., None]]
        for n in range(1, self.Nf+1):
            f += [jnp.cos(2*np.pi*rij/self.L*n), jnp.sin(2*np.pi*rij/self.L*n)]
        return jnp.concatenate(f, axis=-1)

    def _combine(self, h1, h2):
        n = h2.shape[0]
        partitions = [n]

        h2s = jnp.split(h2, partitions, axis=0)
        g2 = [jnp.mean(h, axis=0) for h in h2s if h.size > 0]

        h1s = jnp.split(h1, partitions, axis=0)
        g1 = [jnp.mean(h, axis=0, keepdims=True) for h in h1s if h.size > 0]
        g1 = [jnp.tile(g, [n, 1]) for g in g1]
        f = jnp.concatenate([h1] + g1 + g2, axis=1)
        return f

    def __call__(self, x, t):

        n, dim = x.shape[0], x.shape[1]

        h1 = [jnp.full((n, 1), t)]
        for f in range(1, self.Nf+1):
            h1 += [jnp.full((n, 1), jnp.cos(2*np.pi*t*f)), 
                   jnp.full((n, 1), jnp.sin(2*np.pi*t*f))]
        h1 = jnp.concatenate(h1, axis=-1)
        h2 = self._h2(x)

        for d in range(self.depth-1):

            f = self._combine(h1, h2)

            h1_update = jnp.tanh(self.fc1[d](f))
            h2_update = jnp.tanh(self.fc2[d](h2))

            if d > 0:
                h1 = h1_update + h1
                h2 = h2_update + h2
            else:
                h1 = h1_update 
                h2 = h2_update

        f = self._combine(h1, h2)
        h1 = jnp.tanh(self.fc1[-1](f)) + h1

        final = hk.Linear(dim, w_init=hk.initializers.TruncatedNormal(self.init_stddev), with_bias=False)
        
        #force = jax.grad(softcore)(x, self.L)
        force = jax.grad(self.energy_fn)(x)
        force = jnp.clip(force, a_min = -10.0, a_max = 10.0)
        return -final(h1)*force

def make_ferminet(key, n, dim, depth, h1size, h2size, L, energy_fn):
    x = jax.random.uniform(key, (n, dim), minval=0, maxval=L)
    t = jax.random.uniform(key)

    def forward_fn(x, t):
        net = FermiNet(depth, h1size, h2size, 5, L, energy_fn)
        return net(x.reshape(n, dim), t).reshape(n*dim)
    network = hk.without_apply_rng(hk.transform(forward_fn))
    params = network.init(key, x, t)
    div_fn = lambda params, x, t, key: div(lambda x: network.apply(params, x, t))(key, x)
    return params, network.apply, div_fn
