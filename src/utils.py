import jax
import jax.numpy as jnp
import numpy as np 
import pickle

def get_gr(x, L, bins=100): 
    batchsize, n, dim = x.shape[0], x.shape[1], x.shape[2]
    
    i,j = np.triu_indices(n, k=1)
    rij = (np.reshape(x, (-1, n, 1, dim)) - np.reshape(x, (-1, 1, n, dim)))[:,i,j]
    rij = rij - L*np.rint(rij/L)
    dist = np.linalg.norm(rij, axis=-1) # (batchsize, n*(n-1)/2)
   
    hist, bin_edges = np.histogram(dist.reshape(-1,), range=[0, L/2], bins=bins)
    dr = bin_edges[1] - bin_edges[0]
    hist = hist*2/(n * batchsize)

    rmesh = np.arange(hist.shape[0])*dr
    
    h_id = 4/3*np.pi*n/(L**3)* ((rmesh+dr)**3 - rmesh**3 )
    return rmesh, hist/h_id

def make_pairwise_potential(L): 

    def _yukawa(r):
        '''
        a pretty arbitary soft core potential whose force we added to the learned velocity field to push particles apart 
        '''
        return 2*jnp.exp(-2.5*r)

    def _lj(r):
        r2 = r*r  
        one_R2 = 1.0 / r2
        sig_R2 = one_R2 * 0.3419 * 0.3419
        epairs = 2. * 0.2341 * (jnp.power(sig_R2, 6) - jnp.power(sig_R2, 3))
        return epairs

    f = _lj
    v = lambda r: (f(r) + f(L-r) - 2*f(L/2))*(r<=L/2) + 0.0*(r>L/2)

    def energy_fn(x):
        n, dim = x.shape
        i, j = jnp.triu_indices(n, k=1)
        rij = (jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))[i, j]
        rij = rij - L * jnp.rint(rij/L)
        r = jnp.linalg.norm(rij, axis=-1)
        return jnp.sum(jax.vmap(v)(r))

    return energy_fn 

def divergence_fwd(f):
    def _div_f(x):
        jac = jax.jacfwd(f)
        return jnp.trace(jac(x))
    return _div_f

def divergence_hutchinson(f):
    def _div_f(v, x):
        _, jvp = jax.jvp(f, (x,), (v,))
        return (jvp * v).sum()
    return _div_f

def divergence_scan(f):
    def _div_f(x):
        n = x.shape[0]
        eye = jnp.eye(n)

        def _body_fun(val, i):
            primal, tangent = jax.jvp(f, (x,), (eye[i],))
            return val + tangent[i], None
                                             
        return jax.lax.scan(_body_fun, 0.0, jnp.arange(0, n))[0]
    return _div_f

def divergence_fori(f):
    def _div_f(x):
        n = x.shape[0]
        eye = jnp.eye(n)
        def _body_fun(i, val):
            primal, tangent = jax.jvp(f, (x,), (eye[i],))
            return val + tangent[i]
        return jax.lax.fori_loop(0, n, _body_fun, 0.0)
    return _div_f

def loaddata(filename):
    n = 14
    dim = 3
    dens = 0.016355 
    L = (n/dens)**(1/3)

    with open(filename, "rb") as f:
        data = pickle.load(f)

    X1 = data['x_step']
    X1 = X1.reshape(-1, n, dim)
    X1 = X1[:, :n//2, :] # only take spin up

    print (X1.shape, L)
    X1 -= L * jnp.floor(X1/L)
    
    return X1, n//2, dim, L, 0.0

if __name__=='__main__':
    filename = "/data/zhangqidata/TestHelium3Flow/Helium3FreeFermions_n_14/epoch_000400.pkl"
    #filename = "/data/zhangqidata/TestHelium3Flow/Helium3Jastrow_n_14/epoch_004000.pkl"

    x, n, dim, L, _  = loaddata(filename)

    print (x.shape) 
    print (n, dim, L)

    rmesh, gr = get_gr(x, L, bins=100)

    import matplotlib.pyplot as plt
    plt.plot(rmesh, gr)
    plt.show()

