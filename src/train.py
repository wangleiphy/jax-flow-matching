import jax 
import optax
import haiku as hk

from typing import NamedTuple
import itertools

class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

def train(key, num_epochs, init_params, data, lr):
    
    @jax.jit
    def step(key, i, state, x0, x1):
        key, key_x0, key_t = random.split(key)
        x0 = random.normal(key_x0, x1.shape)
        t = random.uniform(key_t, (args.batchsize,))

        value, grad = value_and_grad(state.params, x0, x1, t)

        updates, opt_state = optimizer.update(grad, state.opt_state)
        params = optax.apply_updates(state.params, updates)

        return TrainingState(params, opt_state), value
    
    optimizer = optax.adam(lr)
    init_opt_state = optimizer.init(init_params)

    state = TrainingState(init_params, init_opt_state)

    itercount = itertools.count()
    for epoch in range(num_epochs):
        key, subkey = jnp.random.split(key)
        data = jax.random.permutation(subkey, data)

        total_loss = 0.0
        for batch_index in range(0, len(X1), args.batchsize):
            q = data[batch_index:batch_index+args.batchsize]
            
            # put momentum to the first half
            key, subkey = jnp.random.split(key)
            p = jax.random.normal(subkey, q.shape)
            x1 = jnp.concatenate([p, q], axis=1)

            key, subkey = jnp.random.split(key)
            state, loss = step(subkey, 
                               next(itercount), 
                               state, 
                               x1)
            total_loss += loss
    
        print (epoch, total_loss/(batch_index+1.))
    return state.params
