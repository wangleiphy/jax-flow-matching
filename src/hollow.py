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
    def transformer(x, h, t):
        mask = jnp.kron(jnp.ones(2), jnp.eye(dim)) # 2 because we do cos/sin
        x_repeat = jnp.repeat(x[None, :], dim, axis=0) # (d, 2d)
        x_repeat = jnp.where(mask, x_repeat, jax.lax.stop_gradient(x_repeat)) 
        y = jnp.concatenate([x_repeat, 
                             h.reshape(dim, -1), 
                             jnp.full((dim, 1), t)
                            ], axis=1) # (d, 2d+h+1)
        return hk.nets.MLP(hidden_sizes+[1], activation=jax.nn.softplus)(y)

    return transformer

def make_hollow_net(key, n, dim, L, nheads, keysize, h1size, h2size, nlayers):

    @hk.without_apply_rng
    @hk.transform
    def hollow_net(x, t): 
        x = x.reshape(n, dim)
        conditioner = make_conditioner(nheads, keysize, dim*h1size)
        transformer = make_transformer([h2size]*nlayers, dim)
        x = jnp.concatenate([jnp.cos(2*jnp.pi*x/L), 
                             jnp.sin(2*jnp.pi*x/L), 
                             ], axis=1) # (n, 2d)
        xt = jnp.concatenate([x, jnp.full((n, 1), t)], axis=1) # (n, 2d+1)
        h = conditioner(xt) # (n, d*h) 
        h = jax.lax.stop_gradient(h)
        out = hk.vmap(transformer, in_axes=(0,0,None), out_axes=0, split_rng=False)(x,h,t) # (n, d)
        return out.reshape(n*dim)

    def divergence_fn(params, x, t, _):
        f = lambda x: hollow_net.apply(params, x, t)
        _, jvp = jax.jvp(f, (x,), (jnp.ones_like(x),))
        return jnp.sum(jvp)

    x = jax.random.uniform(key, (n, dim), minval=0, maxval=L)
    t = jax.random.uniform(key)
    params = hollow_net.init(key, jnp.zeros((n, dim)), 1.0)

    return params, hollow_net.apply, divergence_fn
