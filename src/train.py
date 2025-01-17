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

def train(key, value_and_grad, free_energy_fn, nepoch, batchsize, params, X0, X1, lr, path, L):

    assert (len(X1)%batchsize==0)

    @jax.jit
    def step(key, i, state, x0, x1):
        key, subkey = jax.random.split(key)
        t = jax.random.uniform(subkey, (batchsize,))

        value, grad = value_and_grad(state.params, x0, x1, t)

        updates, opt_state = optimizer.update(grad, state.opt_state)
        params = optax.apply_updates(state.params, updates)

        return TrainingState(params, opt_state), value
    
    optimizer = optax.adam(lr)
    init_opt_state = optimizer.init(params)
    state = TrainingState(params, init_opt_state)

    f = open(os.path.join(path, "loss.txt"), "w", buffering=1, newline="\n")
    g = open(os.path.join(path, "fe.txt"), "w", buffering=1, newline="\n")
    itercount = itertools.count()
    for epoch in range(1, nepoch+1):
        key, subkey = jax.random.split(key)

        X0 = jax.random.permutation(subkey, X0)
        X1 = jax.random.permutation(subkey, X1)

        total_loss = 0.0
        counter = 0 
        for batch_index in range(0, len(X1), batchsize):
            key, subkey = jax.random.split(key)
            state, loss = step(subkey, 
                               next(itercount), 
                               state, 
                               X0[batch_index:batch_index+batchsize],
                               X1[batch_index:batch_index+batchsize]
                               )
            total_loss += loss
            counter += 1

        #print (epoch, total_loss/counter)
        f.write( ("%6d" + "  %.6f" + "\n") % (epoch, total_loss/counter) )

        if epoch % 100 == 0:
            ckpt = {"params": state.params,
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

            key, subkey1, subkey2 = jax.random.split(key,3)
            fe_ub, fe_ub_err, _ = free_energy_fn(subkey1, state.params, batchsize, 1)
            fe_lb, fe_lb_err, _ = free_energy_fn(subkey2, state.params, batchsize, -1)
            g.write( ("%6d" + "  %.6f"*4 + "\n") % (epoch, fe_lb, fe_lb_err, fe_ub, fe_ub_err))

    f.close()
    g.close()
    return state.params

def train2(key, value_and_grad, nepoch, batchsize, params, data, lr, path, L):

    assert (len(data)%batchsize==0)

    @jax.jit
    def step(key, i, state, x1):
        subkeys = jax.random.split(key, x1.shape[0])
        value, grad = value_and_grad(state.params, x1, subkeys)

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
        key, subkey = jax.random.split(key)
        data = jax.random.permutation(subkey, data)

        total_loss = 0.0
        counter = 0 
        for batch_index in range(0, len(data), batchsize):
            x1 = data[batch_index:batch_index+batchsize]

            key, subkey = jax.random.split(key)
            state, loss = step(subkey, next(itercount), state, x1)
            total_loss += loss
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

