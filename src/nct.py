import jax
import jax.numpy as jnp
from jax.experimental import ode

'''
continuous transformation in phase space 
d q /d t =   grad_p H 
d p /d t = - grad_q H 
'''

def phasespace_v(params, hamiltonian_net, x, t):
    p, q = jnp.split(x, 2)
    return jnp.concatenate([-jax.grad(hamiltonian_net, argnums=2)(params, p, q, t), 
                             jax.grad(hamiltonian_net, argnums=1)(params, p, q, t)])

def make_canonical_transformation(hamiltonian_net):

    def canonical_transformation(params, x0, sign):
        assert x0.shape[0]%2 == 0
        assert abs(sign)==1
        def _ode(x, t):    
            v = phasespace_v(params,
                             hamiltonian_net, 
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
    
    return canonical_transformation
