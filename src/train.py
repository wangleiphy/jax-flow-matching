import jax
import jax.numpy as jnp
import optax
import haiku as hk

import checkpoint
import os
from typing import NamedTuple
import itertools

class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

def train(rng, value_and_grad, nepoch, batchsize, params, data, lr, path):

    assert (len(data)%batchsize==0)

    @jax.jit
    def step(rng, i, state, x1):
        sample_rng, rng = jax.random.split(rng)
        x0 = jax.random.normal(sample_rng, x1.shape)
        t = jax.random.uniform(rng, (batchsize,))

        value, grad = value_and_grad(state.params, x0, x1, t)

        updates, opt_state = optimizer.update(grad, state.opt_state)
        params = optax.apply_updates(state.params, updates)

        return TrainingState(params, opt_state), value
    
    optimizer = optax.adam(lr)
    init_opt_state = optimizer.init(params)
    state = TrainingState(params, init_opt_state)

    log_filename = os.path.join(path, "loss.txt")
    f = open(log_filename, "w", buffering=1, newline="\n")
    itercount = itertools.count()
    for epoch in range(1, nepoch+1):
        permute_rng, rng = jax.random.split(rng)
        data = jax.random.permutation(permute_rng, data)

        total_loss = 0.0
        counter = 0 
        for batch_index in range(0, len(data), batchsize):
            x1 = data[batch_index:batch_index+batchsize]

            step_rng, rng = jax.random.split(rng)
            state, d_mean = step(step_rng, next(itercount), state, x1)
            total_loss += d_mean
            counter += 1

        f.write( ("%6d" + "  %.6f" + "\n") % (epoch, total_loss/counter) )

        if epoch % 100 == 0:
            ckpt = {"params": state.params,
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()
    return state.params

