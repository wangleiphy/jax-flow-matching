import jax 
from functools import partial

from mcmc import mcmc_fun

@partial(jax.jit, static_argnums=(1,2,3,4))
def sample_target(key, batchsize, n, dim, logp, mc_epoch=20, mc_steps=100, mc_width=0.05):

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (batchsize, n*dim))
    for _ in range(mc_epoch):
        key, subkey = jax.random.split(key)
        x, acc = mcmc_fun(subkey, logp, x, mc_steps, mc_width)
    return x
