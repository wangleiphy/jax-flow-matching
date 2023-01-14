import jax.numpy as jnp
from functools import partial

@partial(jax.vmap, in_axes=(0, None, None))
def energy_fun(x, n, dim):
    return ... 

def make_free_energy(energy_fun, batched_sampler, n, dim, beta):

    def free_energy(rng, params, sample_size):
        
        x, logp = batched_sampler(rng, params, sample_size)
        e = energy_fun(x, n, dim)

        amount = jnp.exp(- beta * e - logp)
        z, z_err = amount.mean(), amount.std() / jnp.sqrt(x.shape[0])
        lnz, lnz_err = -jnp.log(z)/beta, z_err/(z*beta)
        
        f = e + logp/beta # variational free energy

        return lnz, lnz_err, x, f.mean(), f.std()/jnp.sqrt(x.shape[0])
    
    return free_energy
