'''
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/permutations.html
https://github.com/google/gumbel_sinkhorn/blob/master/sinkhorn_ops.py
'''
import jax
import jax.numpy as jnp 

def log_sinkhorn(log_alpha, n_iter):
    for _ in range(n_iter):
        log_alpha = log_alpha - jax.scipy.special.logsumexp(log_alpha, -1, keepdims=True)
        log_alpha = log_alpha - jax.scipy.special.logsumexp(log_alpha, -2, keepdims=True)
    return jnp.exp(log_alpha)

def gumbel_sinkhorn(key, dist_mat, tau, n_iter, noise_factor): 
    log_alpha = -dist_mat
    gumbel_noise = jax.random.gumbel(key, log_alpha.shape)*noise_factor
    sampled_perm_mat = log_sinkhorn((log_alpha + gumbel_noise)/tau, n_iter)
    print (jnp.sum(sampled_perm_mat, axis=0))
    print (jnp.sum(sampled_perm_mat, axis=1))
    return jnp.argmax(sampled_perm_mat, axis=1)

if __name__=='__main__':
    jax.config.update("jax_enable_x64", True)
    n, dim = 32, 3  
    L = 1.234
    import numpy as np 
    np.set_printoptions(precision=2, suppress=True)
    
    key = jax.random.PRNGKey(42)
    key_x0, key_x1 = jax.random.split(key)
    x0 = jax.random.uniform(key_x0, (n, dim), minval=0, maxval=L)
    x1 = jax.random.uniform(key_x1, (n, dim), minval=0, maxval=L)

    from scipy.optimize import linear_sum_assignment

    def periodic_distance(x0, x1, L):
        '''
        nearest image distance in the box
        '''
        n, dim = x0.shape
        rij = (jnp.reshape(x0, (n, 1, dim)) - jnp.reshape(x1, (1, n, dim)))
        rij = rij - L*jnp.rint(rij/L)
        return jnp.linalg.norm(rij, axis=-1)**2 # (n, n)
    
    dist_mat =periodic_distance(x0, x1, L)
    print (jnp.trace(dist_mat))

    _, col = linear_sum_assignment(dist_mat)
    print (col, jnp.trace(dist_mat[:, col]))
    print (jnp.unique(col).size)
    
    tau = 1e-3
    perm = gumbel_sinkhorn(key, dist_mat, tau=tau, n_iter=100, noise_factor=0)
    print (perm, jnp.trace(dist_mat[:, perm]))
    print (jnp.unique(perm).size)

    from ott.geometry import geometry
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn
    geom = geometry.Geometry(dist_mat, epsilon=tau)
    prob = linear_problem.LinearProblem(geom)
    ot = sinkhorn.Sinkhorn()(prob)
    perm = jnp.argmax(ot.matrix, axis=1)
    print (perm, jnp.trace(dist_mat[:, perm]))
    print (jnp.unique(perm).size)
    print ('Sinkhorn converged:', ot.converged)
