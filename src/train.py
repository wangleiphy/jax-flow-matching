import jax 
import jax.numpy as jnp
import optax
import haiku as hk
import os
import time
from typing import NamedTuple
import itertools

import checkpoint

class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

def train(key, value_and_grad, num_epochs, batchsize, params, data, lr, path):
    
    assert (len(data)%batchsize==0)

    @jax.jit
    def step(key, i, state, x1):
        key, subkey_x0, subkey_t = jax.random.split(key, 3)
        x0 = jax.random.normal(subkey_x0, x1.shape)
        t = jax.random.uniform(subkey_t, (batchsize,))
        
        value, grad = value_and_grad(state.params, x0, x1, t)

        updates, opt_state = optimizer.update(grad, state.opt_state)
        params = optax.apply_updates(state.params, updates)

        return TrainingState(params, opt_state), value
    
    optimizer = optax.adam(lr)
    init_opt_state = optimizer.init(params)
    state = TrainingState(params, init_opt_state)

    time_of_last_ckpt = time.time()
    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w", buffering=1, newline="\n")
    itercount = itertools.count()
    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        data = jax.random.permutation(subkey, data)

        total_loss = 0.0
        counter = 0 
        for batch_index in range(0, len(data), batchsize):
            q = data[batch_index:batch_index+batchsize]
            
            # put momentum to the first half
            key, subkey = jax.random.split(key)
            p = jax.random.normal(subkey, q.shape)
            x1 = jnp.concatenate([p, q], axis=1)

            key, subkey = jax.random.split(key)
            state, loss = step(subkey, 
                               next(itercount), 
                               state, 
                               x1)
            total_loss += loss
            counter += 1
    
        f.write( ("%6d" + "  %.6f" + "\n") % (epoch, total_loss/counter) )

        if time.time() - time_of_last_ckpt > 600:
            ckpt = {"params": state.params,
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)
            time_of_last_ckpt = time.time()

    f.close()
    return state.params
