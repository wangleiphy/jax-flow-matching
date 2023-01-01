import jax 
from jax.config import config
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.scipy.stats import norm
import matplotlib.pyplot as plt

def make_logp(energy_fun, n, dim, beta):

    def logp(x, lam):
        return lam * -beta*energy_fun(x, n, dim) + (1-lam)*(norm.logpdf(x).sum((1,2)))
    return logp

@partial(jax.jit, static_argnums=0)

def mcmc(logp_fn, x_init, key, mc_steps, mc_width=0.02):
    """
        Markov Chain Monte Carlo sampling algorithm.

    INPUT:
        logp_fn: callable that evaluate log-probability of a batch of configuration x.
            The signature is logp_fn(x), where x has shape (batch, n, dim).
        x_init: initial value of x, with shape (batch, n, dim).
        key: initial PRNG key.
        mc_steps: total number of Monte Carlo steps.
        mc_width: size of the Monte Carlo proposal.

    OUTPUT:
        x: resulting batch samples, with the same shape as `x_init`.
    """
    def step(i, state):
        x, logp, key, num_accepts = state
        key, key_proposal, key_accept = jax.random.split(key, 3)
        
        x_proposal = x + mc_width * jax.random.normal(key_proposal, x.shape)
        logp_proposal = logp_fn(x_proposal)

        ratio = jnp.exp((logp_proposal - logp))
        accept = jax.random.uniform(key_accept, ratio.shape) < ratio

        x_new = jnp.where(accept[:, None, None], x_proposal, x)
        logp_new = jnp.where(accept, logp_proposal, logp)
        num_accepts += accept.sum()
        return x_new, logp_new, key, num_accepts
    
    logp_init = logp_fn(x_init)

    x, logp, key, num_accepts = jax.lax.fori_loop(0, mc_steps, step, (x_init, logp_init, key, 0.))
    return x

if __name__ == "__main__":
    config.update("jax_enable_x64", True)
    import argparse
    parser = argparse.ArgumentParser(description='')

    group = parser.add_argument_group('sample parameters')
    group.add_argument('-batchsize', type=int, default=8192, help='')
    group.add_argument('-mc', type=int, default=20, help='')
    group.add_argument('-anneal', type=int, default=100, help='')

    group = parser.add_argument_group('physics parameters')
    group.add_argument('-n', type=int, default=6, help='The number of particles')
    group.add_argument('-dim', type=int, default=2, help='The dimensions of the system')
    group.add_argument('-beta', type=float, default=10.0, help='')

    args = parser.parse_args()

    from energy import energy_fun
    logp_fun = make_logp(energy_fun, args.n, args.dim, args.beta)

    rng = jax.random.PRNGKey(42)

    #compute the partition function via AIS https://arxiv.org/abs/physics/9803008
    lams = jnp.linspace(0, 1, args.anneal)
    x = jax.random.normal(rng, (args.batchsize, args.n, args.dim))
    w = logp_fun(x, lams[1]) - logp_fun(x, lams[0])

    for j in range(1, args.anneal-1):
        rng, sub_rng = jax.random.split(rng)
        x = mcmc(lambda x: logp_fun(x, lams[j]), x, sub_rng, args.mc)
        w += logp_fun(x, lams[j+1]) - logp_fun(x, lams[j])
    
    z, z_err = jnp.mean(jnp.exp(w)), jnp.std(jnp.exp(w))/jnp.sqrt(args.batchsize)

    lnz, lnz_err = jnp.log(z), z_err/z # error propagation through ln function

    print (-lnz/args.beta, '+/-', lnz_err)
    
    # draw
    
    x = jnp.reshape(x, (args.batchsize*args.n, args.dim))


    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    H, xedges, yedges = np.histogram2d(x[:, 0], x[:, 1], 
                                        bins=100, 
                                        range=((-4, 4), (-4, 4)),
                        density=True)
    plt.imshow(H, interpolation="nearest", 
                extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
                cmap="inferno")

    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.savefig('ais_n%i_dim%i_beta%f.png' % (args.n, args.dim, args.beta))
    