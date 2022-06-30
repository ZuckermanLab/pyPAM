import parallel_affine as pa
import numpy as np
import time
import emcee 
import multiprocessing as mp
import matplotlib.pyplot as plt


# using example from: https://emcee.readthedocs.io/en/stable/tutorials/quickstart/#quickstart
def log_prob(x, mu, cov):
    diff = x - mu
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))


if __name__ == "__main__": 

    # using example from: https://emcee.readthedocs.io/en/stable/tutorials/quickstart/#quickstart
    np.random.seed(42)

    # example model parameters
    n_dim = 5
    means = np.random.rand(n_dim)
    cov = 0.5 - np.random.rand(n_dim**2).reshape((n_dim, n_dim))
    cov = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)
   
    # sampling parameters 
    n_ensembles = 4
    n_walkers = 10
    n_steps = 100
    n_final_steps = 1000
    n_mixing_steps = 10
    log_prob_args = [means, cov]
    n_cores = n_ensembles

    # initial start points w/ shape = (n_ensembles, n_walkers, n_dim)
    p_0 = np.array([np.random.rand(n_walkers, n_dim) for i in range(n_ensembles)])

    # run parallel sampler (w/ mixing initialization) and time it
    t0 = time.time()
    sampler = pa.ParallelEnsembleSampler(n_ensembles, n_walkers, n_dim, log_prob, log_prob_args)
    states = sampler.run_mixing_sampler(p_0, n_steps, n_cores, n_mixing_steps, n_final_steps)  
    samples = sampler.get_flat_samples()
    t = time.time() - t0
    n_log_prob_calls = n_ensembles*n_walkers*n_steps*n_mixing_steps + n_ensembles*n_walkers*n_final_steps
    print(f'{n_cores} core(s), w/ mixing: {t} sec wall-clock for {n_log_prob_calls} log probability calculations --> {n_log_prob_calls/t} calculations/sec')

    # run parallel sampler (w/o mixing initialization) and time it
    n_steps_total = n_steps*n_mixing_steps + n_final_steps
    t0 = time.time()
    sampler = pa.ParallelEnsembleSampler(n_ensembles, n_walkers, n_dim, log_prob, log_prob_args)
    states = sampler.run_sampler(p_0, n_steps_total, n_cores)  # same total number of log probability calculations as above example
    samples_2 = sampler.get_flat_samples()
    t = time.time() - t0
    n_log_prob_calls = n_ensembles*n_walkers*n_steps*n_mixing_steps + n_ensembles*n_walkers*n_final_steps
    print(f'{n_cores} core(s), w/o mixing: {t} sec wall-clock for {n_log_prob_calls} log probability calculations --> {n_log_prob_calls/t} calculations/sec')

    # run parallel EMCEE sampler (reference) and time it
    n_walkers_emcee = n_walkers*n_ensembles
    n_steps_emcee = n_steps*n_mixing_steps + n_final_steps
    p_0_emcee = np.random.rand(n_walkers_emcee, n_dim)
    t0 = time.time()
    with mp.Pool(n_cores) as pool:
        sampler_emcee = emcee.EnsembleSampler(n_walkers_emcee, n_dim, log_prob, args=log_prob_args, pool=pool)
        sampler_emcee.run_mcmc(p_0_emcee,n_steps_emcee)
    samples_emcee_p = sampler_emcee.get_chain(flat=True)
    t = time.time() - t0
    n_log_prob_calls = n_walkers_emcee*n_steps_emcee
    print(f'{n_cores} core(s), w/o mixing: {t} sec wall-clock for {n_log_prob_calls} log probability calculations --> {n_log_prob_calls/t} calculations/sec')

    # plot 1d marginal posteriors
    fig, axs = plt.subplots(n_dim, figsize=(10,10))
    plt.suptitle('1d marginal posterior distributions')
    for i, ax in enumerate(axs.flatten()):
        ax.hist(np.transpose(samples)[i], 100, color="blue", histtype="step", density=True, label='parallel w/ mixing')
        ax.hist(np.transpose(samples_2)[i], 100, color="green", histtype="step", density=True, label='parallel w/o mixing')
        ax.hist(np.transpose(samples_emcee_p)[i], 100, color="red", histtype="step", density=True, label='parallel EMCEE')
        ax.legend()
    plt.savefig('parallel_affine_example_dist.png')