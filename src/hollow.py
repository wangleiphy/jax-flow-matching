'''
hollow net with cheap divergence operator https://arxiv.org/abs/1912.03579
in addition, we make this permutation equvariant 
'''
import jax
import jax.numpy as jnp
import haiku as hk

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

def make_transformer(hidden_sizes, dim):
    def transformer(x, h):
        mask = jnp.kron(jnp.ones(2), jnp.eye(dim)) # 2 because we do cos/sin
        x_repeat = jnp.repeat(x[None, :], dim, axis=0) # (d, 2d)
        x_repeat = jnp.where(mask, x_repeat, jax.lax.stop_gradient(x_repeat)) 
        y = jnp.concatenate([x_repeat, h.reshape(dim, -1)], axis=1) # (d, 2d+h)
        return hk.nets.MLP(hidden_sizes+[1], activation=jax.nn.softplus)(y)

    return transformer

def make_hollow_net(hidden_sizes, L):

    @hk.without_apply_rng
    @hk.transform
    def hollow_net(x): 
        n, dim = x.shape
        conditioner = make_conditioner(8, 16, dim*16)
        transformer = make_transformer(hidden_sizes, dim)
        x = jnp.concatenate([jnp.cos(2*jnp.pi*x/L), 
                             jnp.sin(2*jnp.pi*x/L), 
                             ], axis=1) # (n, 2d)
        h = conditioner(x) # (n, d*h) 
        h = jax.lax.stop_gradient(h)
        return hk.vmap(transformer, split_rng=False)(x,h) # (n, d)

    return hollow_net

def make_divergence_fn(network):

    def divergence_fn(params, x):
        f = lambda x: network.apply(params, x)
        _, jvp = jax.jvp(f, (x,), (jnp.ones_like(x),))
        return jnp.sum(jvp)

    return divergence_fn
