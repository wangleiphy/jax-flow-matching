import jax
import jax.numpy as jnp
from jax.experimental import ode

'''
continuous point transformation in phase space 
d q /d t = v 
d p /d t = - p*(dv/dq)
'''

def phasespace_v(params, vec_field_net, x, t):
    p, q = jnp.split(x, 2)
    v, vjp = jax.vjp(lambda _: vec_field_net(params, _, t), q)
    u, = vjp(p)
    return jnp.concatenate([-u, v])

def make_point_transformation(vec_field_net):

    def point_transformation(params, x0, sign):
        assert x0.shape[0]%2 == 0
        assert abs(sign)==1
        def _ode(x, t):    
            v = phasespace_v(params, 
                            vec_field_net, 
                            x, 
                            t if sign==1 else 1.0-t
                            )
            return sign*v

        x1 = ode.odeint(_ode,
                        x0,
                        jnp.array([0.0, 1.0]),
                        rtol=1e-10, atol=1e-10,
                        mxstep=5000
                        )
        return x1[-1]
    
    return point_transformation
