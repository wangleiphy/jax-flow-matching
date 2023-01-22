import jax
import jax.numpy as jnp
import numpy as np 

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

def softcore(x, L):
    '''
    a pretty arbitary soft core potential whose force we added to the learned velocity field to push particles apart 
    '''
    n, dim = x.shape
    i, j = jnp.triu_indices(n, k=1)
    rij = (jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))[i, j]
    rij = rij - L * jnp.rint(rij/L)
    r = jnp.linalg.norm(rij, axis=-1)

    _f = lambda r: 2*jnp.exp(-2.5*r)
    _v = lambda r: (_f(r) + _f(L-r) - 2*_f(L/2))*(r<=L/2) + 0.0*(r>L/2)

    return jnp.sum(jax.vmap(_v)(r))

def divergence_fwd(f):
    def _div_f(x):
        jac = jax.jacfwd(f)
        return jnp.trace(jac(x))
    return _div_f

def divergence_hutchinson(f):
    def _div_f(key, x):
        v = jax.random.normal(key, x.shape)
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

def loaddata(dataset):

    data = jnp.load(dataset)
    X1 = data['positions']
    L = data['cell_vectors'][0,0]
    T = data['T']
    n, dim = X1.shape[1], X1.shape[2]
    X1 = X1.reshape(-1, n*dim)
    assert (n == data['N'])

    #data = np.loadtxt(args.dataset)
    #datasize, n, dim = 1000, 64, 3
    #L = data[-1, -1]
    #X1 = data[:, :-3]
    #X1 = X1.reshape(datasize, n*dim)

    print (X1.shape, L)
    print (jnp.min(X1), jnp.max(X1))
    X1 -= L * jnp.floor(X1/L)
    
    return X1, n, dim, L, T
