import jax.numpy as jnp
from jax import vmap
from functools import partial

@partial(vmap, in_axes=(0, None, None))
def energy_fun(x, n, dim):
    i, j = jnp.triu_indices(n, k=1)
    r_ee = jnp.linalg.norm((jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))[i,j], axis=-1)
    v_ee = jnp.sum(1/r_ee)
    return jnp.sum(x**2) + v_ee

def make_free_energy(energy_fun, batched_sampler, logp_fun, n, dim, beta):

    def free_energy(rng, params, sample_size):
        
        x = batched_sampler(rng, params, sample_size)
        e = energy_fun(x, n, dim)
        logp = logp_fun(params, x)

        amount = jnp.exp(- beta * e - logp)
        z, z_err = amount.mean(), amount.std() / jnp.sqrt(x.shape[0])
        lnz, lnz_err = -jnp.log(z)/beta, z_err/(z*beta)
        
        f = e + logp/beta # variational free energy

        return lnz, lnz_err, x, f.mean(), f.std()/jnp.sqrt(x.shape[0])
    
    return free_energy
