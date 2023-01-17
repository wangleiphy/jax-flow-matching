'''
hallow net with cheap divergence operator https://arxiv.org/abs/1912.03579
in addition, we make this permutation equvariant 
'''

import jax
import jax.numpy as jnp
import haiku as hk

from maskedlinear import MaskedLinear

def make_conditioner(num_heads, key_size, model_size):
    def conditioner(x):
        mask = 1-jnp.eye(x.shape[0])
        mask = mask[None, :, :] 
        mha = hk.MultiHeadAttention(num_heads=num_heads,
                                    key_size=key_size,
                                    model_size=model_size,
                                    w_init=hk.initializers.TruncatedNormal(stddev=0.01),
                                   )

        return mha(x, x, x, mask=mask)
    return conditioner

def make_transformer(hidden_sizes):
    def transformer(x, h):
        dim = x.shape[0]
        mask = jnp.eye(dim, dtype=bool)
        x_repeat = jnp.repeat(x[None, :], dim, axis=0) # (d, d)
        x_repeat = jnp.where(mask, x_repeat, jax.lax.stop_gradient(x_repeat)) 
        y = jnp.concatenate([x_repeat, h.reshape(dim, -1)], axis=1) # (d, d+h)
        return hk.nets.MLP(hidden_sizes+[1], activation=jax.nn.softplus)(y)

    return transformer

def make_hallow_net(hidden_sizes):

    @hk.without_apply_rng
    @hk.transform
    def hallow_net(x): 
        n, dim = x.shape
        conditioner = make_conditioner(8, 16, dim*16)
        transformer = make_transformer(hidden_sizes)
        h = conditioner(x) # (n, d*h) 
        h = jax.lax.stop_gradient(h)
        return hk.vmap(transformer, split_rng=False)(x,h) # (n, dim)

    return hallow_net

def make_divergence_fn(network, n, dim):

    def divergence_fn(params, x):
        f = lambda x: network.apply(params, x.reshape(n, dim)).reshape(-1)
        x = x.reshape(-1)
        _, jvp = jax.jvp(f, (x,), (jnp.ones_like(x),))
        return jnp.sum(jvp)

    return divergence_fn
