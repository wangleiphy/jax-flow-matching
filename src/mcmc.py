import jax.numpy as jnp
from jax import random, lax, jit
from functools import partial

@partial(jit, static_argnums=1)
def mcmc_fun(rng, logp_fn, x_init, mc_steps, mc_width=0.02):
    """
        Markov Chain Monte Carlo sampling algorithm.
    INPUT:
        logp_fn: callable that evaluate log-probability of a batch of configuration x.
            The signature is logp_fn(x), where x has shape (batch, n*dim).
        x_init: initial value of x, with shape (batch, n*dim).
        key: initial PRNG key.
        mc_steps: total number of Monte Carlo steps.
        mc_width: size of the Monte Carlo proposal.
    OUTPUT:
        x: resulting batch samples, with the same shape as `x_init`.
    """
    def step(i, state):
        rng, x, logp, num_accepts = state
        rng, proposal_rng, accept_rng = random.split(rng, 3)
        
        x_proposal = x + mc_width * random.normal(proposal_rng, x.shape)
        logp_proposal = logp_fn(x_proposal)
        ratio = jnp.exp((logp_proposal - logp))
        accept = random.uniform(accept_rng, ratio.shape) < ratio
        x_new = jnp.where(accept[:, None], x_proposal, x)
        logp_new = jnp.where(accept, logp_proposal, logp)
        num_accepts += accept.sum()
        return rng, x_new, logp_new, num_accepts
    
    logp_init = logp_fn(x_init)

    rng, x, logp, num_accepts = lax.fori_loop(0, mc_steps, step, (rng, x_init, logp_init, 0.))
    batch = x.shape[0]
    accept_rate = num_accepts / (mc_steps * batch)
    return x, accept_rate