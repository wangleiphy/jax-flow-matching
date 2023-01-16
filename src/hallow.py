'''
hallow net with cheap divergence operator https://arxiv.org/abs/1912.03579
in addition, we make this permutation equvariant 
'''

import jax
import jax.numpy as jnp
import haiku as hk

from utils import divergence_fwd

def make_conditioner(num_heads, key_size, model_size):
    def conditioner(x):
        mha = hk.MultiHeadAttention(num_heads=num_heads,
                                    key_size=key_size,
                                    model_size=model_size,
                                    w_init=hk.initializers.TruncatedNormal(stddev=0.01),
                                   )

        return jnp.sum(mha(x, x, x), axis=0) # (n-1, h) -> (h,)
    return conditioner

def make_transformer(hidden_sizes):
    def transformer(x, h):
        dim = x.shape[0]
        xh = jnp.concatenate([x, h], axis=0) # (dim+h,)
        return hk.nets.MLP(hidden_sizes+[dim], activation=jax.nn.softplus)(xh)
    return transformer

def make_hallow_net(hidden_sizes):

    @hk.without_apply_rng
    @hk.transform
    def hallow_net(x): 
        conditioner = make_conditioner(8, 16, 16)
        transformer = make_transformer(hidden_sizes)

        n, dim = x.shape
        mask = jnp.eye(n, dtype=bool)
        x_repeat = jnp.repeat(x[None, :, :], n, axis=0) # (n, n, dim)
        x_hallow = x_repeat[~mask].reshape(n, (n-1), dim) # (n, (n-1), dim)
    
        h = hk.vmap(conditioner, split_rng=False)(x_hallow) # (n, h) 

        def div(x, h):
            return divergence_fwd(lambda x: transformer(x, h))(x)
        return hk.vmap(transformer, split_rng=False)(x, h), \
               jnp.sum(hk.vmap(div, split_rng=False)(x, h))

    return hallow_net
