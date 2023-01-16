import jax
import jax.numpy as jnp
from jax.experimental import ode
from functools import partial
from utils import divergence_fori as div

def make_flow(vec_field_net, dim, L, mxstep=100):

    def base_logp(x):
        return -dim*jnp.log(L)

    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def forward(params, x0):
        def _ode(x, t):
            return vec_field_net(params, x, t)
        
        xt = ode.odeint(_ode,
                 x0,
                 jnp.linspace(0, 1, 5), 
                 rtol=1e-10, atol=1e-10,
                 mxstep=mxstep
                 )
        return xt
    
    @partial(jax.vmap, in_axes=(None, 0), out_axes=(0,0))
    def forward_with_logp(params, x0):
        def _ode(state, t):
            x = state[0]  
            return vec_field_net(params, x, t), \
                 - div(lambda x: vec_field_net(params, x, t))(x)
        
        logp0 = base_logp(x0)

        xt, logpt = ode.odeint(_ode,
                 [x0, logp0],
                 jnp.array([0.0, 1.0]),
                 rtol=1e-10, atol=1e-10,
                 mxstep=mxstep
                 )
        return xt[-1], logpt[-1]

    @partial(jax.jit, static_argnums=2)
    def sample_and_logp_fn(key, params, batchsize):
        x0 = jax.random.uniform(key, (batchsize, dim), minval=0, maxval=L)
        return forward_with_logp(params, x0)

    @partial(jax.jit, static_argnums=2)
    def sample_fn(key, params, batchsize):
        x0 = jax.random.uniform(key, (batchsize, dim), minval=0, maxval=L)
        return forward(params, x0)
    
    return sample_fn, sample_and_logp_fn
