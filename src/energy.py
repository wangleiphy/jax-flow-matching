import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import mdtraj as md
import numba as nb
from numba import njit, float64, int64
from numba.typed import List
from numba.types import ListType, UniTuple


@nb.njit(float64(float64[:], float64[:], float64[:], int64[:]))
def distance_pbc(vi, vj, box, shift):
    vjp = vj + box * shift
    return np.linalg.norm(vi - vjp)


@nb.njit(int64[:, :, :](float64[:, :], float64[:], float64, int64))
def compute_neighborlist_inner(positions, box, cutoff, max_neighbor):
    natoms = positions.shape[0]
    shift = np.array([0, 0, 0], dtype=np.int64)
    cutoff_array = np.ones((3, )) * cutoff

    pairs = np.zeros((natoms, max_neighbor, 5), dtype=np.int64) + natoms
    dists = np.zeros((natoms, max_neighbor), dtype=np.float64) + cutoff + 10.0
    for ii in range(natoms):
        vi_incell = positions[ii] % box
        vmax = vi_incell + cutoff_array
        vmin = vi_incell - cutoff_array

        nx_max, ny_max, nz_max = ((vi_incell + cutoff_array) // box).astype(
            np.int64)
        nx_min, ny_min, nz_min = ((vi_incell - cutoff_array) // box).astype(
            np.int64)

        for jj in range(ii + 1, natoms):
            vj_incell = positions[jj] % box
            for nx in range(nx_min, nx_max + 1):
                for ny in range(ny_min, ny_max + 1):
                    for nz in range(nz_min, nz_max + 1):
                        shift[0] = nx
                        shift[1] = ny
                        shift[2] = nz
                        dist = distance_pbc(vi_incell, vj_incell, box, shift)
                        if dist < cutoff:
                            # add ii
                            imax = np.argmax(dists[ii, :])
                            if dist < dists[ii, imax]:
                                dists[ii, imax] = dist
                                pairs[ii, imax, 0] = ii
                                pairs[ii, imax, 1] = jj
                                pairs[ii, imax, 2] = nx
                                pairs[ii, imax, 3] = ny
                                pairs[ii, imax, 4] = nz
                            # add jj
                            jmax = np.argmax(dists[jj, :])
                            if dist < dists[jj, jmax]:
                                dists[jj, jmax] = dist
                                pairs[jj, jmax, 0] = jj
                                pairs[jj, jmax, 1] = ii
                                pairs[jj, jmax, 2] = -nx
                                pairs[jj, jmax, 3] = -ny
                                pairs[jj, jmax, 4] = -nz
    return pairs


def compute_neighborlist(positions, box, cutoff=0.3419 * 3, max_neighbor=160):
    pairs = np.array(
        compute_neighborlist_inner(np.array(positions), np.array(box), cutoff,
                                   max_neighbor)).reshape((-1, 5))
    return jnp.array(pairs)


def generate_energy_function(function):
    @jax.jit
    def energy_function(positions, box, pairs):
        pos_center = positions % box

        mask = jnp.piecewise(
            pairs[:, 0] - pairs[:, 1],
            (pairs[:, 0] - pairs[:, 1] != 0, pairs[:, 0] - pairs[:, 1] == 0),
            (lambda x: jnp.array(1), lambda x: jnp.array(0)))

        pos0 = pos_center[pairs[:, 0], :]
        pos1 = pos_center[pairs[:, 1], :]
        shift = pairs[:, 2:] * box
        r_vec = (pos1 + shift) - pos0
        dist = jnp.linalg.norm(r_vec, axis=1)
        r2 = dist * dist
        return (function(r2) * mask).sum()

    return energy_function

@jax.jit
def lennard_jones_energy(r2):
    one_R2 = 1.0 / r2
    sig_R2 = one_R2 * 0.3419 * 0.3419
    epairs = 2. * 0.9794744 * (jnp.power(sig_R2, 6) - jnp.power(sig_R2, 3))
    return epairs

@jax.jit
def wca_potential_energy(r2):
    sigma = 0.3419
    epsilon = 0.9794744
    cutoff = (2 ** (1/6)) * sigma
    energy_shift = epsilon / 2

    one_R2 = 1.0 / r2
    sig_R2 = one_R2 * sigma * sigma
    epairs = (2. * epsilon * (jnp.power(sig_R2, 6) - jnp.power(sig_R2, 3)) + energy_shift) * (r2 <= cutoff ** 2)

    return epairs

lj_efunc = generate_energy_function(lennard_jones_energy)
wca_efunc = generate_energy_function(wca_potential_energy)


def jax_nblist(pos, box):
    return jax.pure_callback(compute_neighborlist,
                             jax.ShapeDtypeStruct((160 * pos.shape[-2], 5),
                                                  np.int64),
                             pos,
                             box,
                             vectorized=False)


jax_nblist = jax.custom_jvp(jax_nblist)


@jax_nblist.defjvp
def _jax_nblist_jvp(primals, tangents):
    val = jax_nblist(*primals)
    grad = jnp.zeros(val.shape, dtype=np.int64)
    return val, grad

def make_energy(n, dim, L):
    def energy_fn(x0, x1):
        box = jnp.ones((dim, )) * L
        e0 = wca_efunc(x0, box, jax_nblist(x0, box))
        e1 = lj_efunc(x1, box, jax_nblist(x1, box))
        return e1 - e0
    return energy_fn

def make_free_energy(batched_sampler, energy_fn, n, dim, L, T):

    kT = 8.314463e-3*T
    print ('kBT', kT)
    
    @partial(jax.jit, static_argnums=(2, 3))
    def free_energy_bound(key, params, batchsize, sign):
        '''
        upper bound sign = 1
        lower bound sign = -1
        '''
        x0, x1, logp = batched_sampler(key, params, batchsize, sign)
        x0 -= L * jnp.floor(x0 / L)
        x1 -= L * jnp.floor(x1 / L)
        e = jax.vmap(energy_fn)(x0.reshape(batchsize, n, dim),
                                x1.reshape(batchsize, n, dim))
        f = e + logp * kT
        return f.mean(), f.std() / jnp.sqrt(batchsize)

    return free_energy_bound
