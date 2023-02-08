import jax
import jax.numpy as jnp
from jax.experimental import ode
from functools import partial

def make_flow(vec_field_net, div_fn, X0, X1, mxstep=1000):
    
    def sample_x0(key, batchsize):
        idx = jax.random.choice(key, jnp.arange(len(X0)), (batchsize,), replace=False)
        return X0[idx]

    def sample_x1(key, batchsize):
        idx = jax.random.choice(key, jnp.arange(len(X1)), (batchsize,), replace=False)
        return X1[idx]

    @partial(jax.vmap, in_axes=(None, 0, None), out_axes=0)
    def integrate(params, x0, sign):
        def _ode(x, t):
            return sign*vec_field_net(params, x, t if sign==1 else 1-t)
        
        xt = ode.odeint(_ode,
                 x0,
                 jnp.linspace(0, 1, 5), 
                 rtol=1e-10, atol=1e-10,
                 mxstep=mxstep
                 )
        return xt
    
    @partial(jax.vmap, in_axes=(None, 0, 0, None), out_axes=(0,0,0))
    def integrate_with_logp(params, x0, v, sign):
        def _ode(state, t):
            x = state[0]  
            time = t if sign==1 else 1-t
            return sign*vec_field_net(params, x, time), \
                  -sign*div_fn(params, x, time, v)
        
        logp0 = 0.0
        xt, logpt = ode.odeint(_ode,
                 [x0, logp0],
                 jnp.array([0.0, 1.0]),
                 rtol=1e-10, atol=1e-10,
                 mxstep=mxstep
                 )
        return xt[0], xt[1], sign*logpt[-1]

    @partial(jax.jit, static_argnums=(2,3))
    def sample_and_logp_fn(key, params, batchsize, forward):
        key1, key2 = jax.random.split(key)
        if forward:
            x = sample_x0(key1, batchsize)
        else:
            x = sample_x1(key1, batchsize)
        v = jax.random.normal(key2, x.shape)
        return integrate_with_logp(params, x, v, 1 if forward else -1)

    @partial(jax.jit, static_argnums=(2, 3))
    def sample_fn(key, params, batchsize, forward):
        if forward:
            x = sample_x0(key, batchsize)
        else:
            x = sample_x1(key, batchsize)
        return integrate(params, x, 1 if forward else -1)

    return sample_fn, sample_and_logp_fn
