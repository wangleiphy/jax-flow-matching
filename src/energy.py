import jax
import jax.numpy as jnp
from functools import partial

def make_energy(n, dim):

    def kinetic(p):
        return jnp.sum(p**2)/2

    def potential(q):
        i, j = jnp.triu_indices(n, k=1)
        r_ee = jnp.linalg.norm((jnp.reshape(q, (n, 1, dim)) - jnp.reshape(q, (1, n, dim)))[i,j], axis=-1)
        v_ee = jnp.sum(1/r_ee)
        return jnp.sum(q**2) + v_ee
    
    def energy(x):
        assert x.size == 2*n*dim
        p, q = jnp.split(x, 2)
        return kinetic(p) + potential(q)

    return jax.vmap(energy), jax.vmap(potential)

def make_free_energy(energy_fun, sampler, n, dim, beta):

    def free_energy(key, params, batchsize):
        
        x, logp = sampler(key, params, batchsize)
        e = energy_fun(x)

        amount = jnp.exp(- beta * e - logp)
        z, z_err = amount.mean(), amount.std() / jnp.sqrt(x.shape[0])
        lnz, lnz_err = -jnp.log(z)/beta, z_err/(z*beta)
        
        f = e + logp/beta # variational free energy

        return lnz, lnz_err, x, f.mean(), f.std()/jnp.sqrt(x.shape[0])
    
    return free_energy
