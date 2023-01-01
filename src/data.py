from jax import random
from mcmc import mcmc_fun
from energy import energy_fun

from sklearn import datasets, preprocessing

def make_sampler(data_size: int, name: str ='mcmc'):

    if name == 'mcmc':

        def sampler(rng, beta, n, dim, mc_epoch=20, mc_steps=100, mc_width=0.05):
            
            sample_rng, rng = random.split(rng)
            X = random.normal(sample_rng, (data_size, n*dim))
            X0 = X

            for _ in range(mc_epoch):
                mcmc_rng, rng = random.split(rng)
                X, acc = mcmc_fun(mcmc_rng, lambda X: -beta*energy_fun(X, n, dim), X, mc_steps, mc_width)
                e = energy_fun(X, n, dim)
            
            X1 = X

            return X0, X1

    if name == 'moons':

        def sampler(rng, dim):
            
            X0 = random.normal(rng, (data_size, dim))
            
            scaler = preprocessing.StandardScaler()
            X, _ = datasets.make_moons(n_samples=data_size, noise=.05)
            
            X1 = scaler.fit_transform(X)

            return X0, X1

    if name == 'gaussian':

        def sampler(rng, dim, mu=0.5, sigma=0.8):

            X0 = random.normal(rng, (data_size, dim))
            X1 = X0*sigma +mu

            return X0, X1

    return sampler

if __name__ == '__main__':
    sampler = make_sampler(20, 'moons')
    X0, X1 = sampler(random.PRNGKey(42), 2)
    print(X0.shape)

