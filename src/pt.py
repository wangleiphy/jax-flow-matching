import jax
import jax.numpy as jnp
from jax.experimental import ode

'''
continuous point transformation in phase space 
d q /d t = v 
d p /d t = - p*(dv/dq)
'''

def PointTransformation(vec_field_net):

    def forward(params, x0):
        assert x0.shape[0]%2 == 0
        def _ode(x, t):    
            p, q = jnp.split(x, 2)
            v, vjp = jax.vjp(lambda _: vec_field_net(params, _, t), q)
            u, = vjp(p)
            return jnp.concatenate([-u, v])

        x1 = ode.odeint(_ode,
                        x0,
                        jnp.array([0.0, 1.0]),
                        rtol=1e-10, atol=1e-10,
                        mxstep=5000
                        )
        return x1[-1]
    
    def reverse(params, x1):
        assert x1.shape[0]%2 == 0
        def _ode(x, t):
            p, q = jnp.split(x, 2)
            v, vjp = jax.vjp(lambda _: vec_field_net(params, _, 1.0-t), q)
            u, = vjp(p)
            return -jnp.concatenate([-u, v])
        
        x0 = ode.odeint(_ode,
                        x1, 
                        jnp.array([0.0, 1.0]),
                        rtol=1e-10, atol=1e-10,
                        mxstep=5000
                       )
        return x0[-1]

    return forward, reverse
