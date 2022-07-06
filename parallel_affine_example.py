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
    n_steps = 200
    n_final_steps = 1000
    n_mixing_steps = 10
    log_prob_args = [means, cov]
    n_cores = n_ensembles
    thin = 10

    # initial start points w/ shape = (n_ensembles, n_walkers, n_dim)
    p_0 = np.array([np.random.rand(n_walkers, n_dim) for i in range(n_ensembles)])

    # set backend filenames
    backend_fnames = [f'PAIES_backend_test_{i}.h5' for i in range(n_ensembles)]

    # move set
    moves = [[
        (emcee.moves.StretchMove(), 0.8),
        (emcee.moves.DESnookerMove(), 0.2),
    ] for i in range(n_ensembles)]

    # run parallel sampler (w/ mixing initialization) and time it
    ### note: currently the samples are discarded during the 'mixing' stages, 
    ### and only the samples from the final stage are kept
    ### also note: the mixing procedure currently requires an 'empty' backend to start with
    t0 = time.time()
    sampler = pa.ParallelEnsembleSampler(n_ensembles, n_walkers, n_dim, log_prob, log_prob_args, thin, backend_fnames, moves)
    sampler.reset_backend()  # warning: removes samples from existing backend file
    states = sampler.run_mixing_sampler(p_0, n_steps, n_cores, n_mixing_steps, n_final_steps,thin)  
    samples = sampler.get_flat_samples()
    print(np.shape(samples))
    t = time.time() - t0
    n_log_prob_calls = n_ensembles*n_walkers*n_steps*n_mixing_steps + n_ensembles*n_walkers*n_final_steps
    print(f'{n_cores} core(s), w/ mixing: {t} sec wall-clock for {n_log_prob_calls} log probability calculations --> {n_log_prob_calls/t} calculations/sec')
    sampler.print_backend_sampling_summary()
    sampler.reset_backend()

    # run parallel sampler (w/o mixing initialization) and time it
    n_steps_total = n_steps*n_mixing_steps + n_final_steps
    t0 = time.time()
    sampler = pa.ParallelEnsembleSampler(n_ensembles, n_walkers, n_dim, log_prob, log_prob_args, thin, backend_fnames, moves)
    states = sampler.run_sampler(p_0, n_steps_total, n_cores, thin)  # same total number of log probability calculations as above example
    samples_2 = sampler.get_flat_samples()
    print(np.shape(samples_2))
    t = time.time() - t0
    n_log_prob_calls = n_ensembles*n_walkers*n_steps*n_mixing_steps + n_ensembles*n_walkers*n_final_steps
    print(f'{n_cores} core(s), w/o mixing: {t} sec wall-clock for {n_log_prob_calls} log probability calculations --> {n_log_prob_calls/t} calculations/sec')
    sampler.print_backend_sampling_summary()
    sampler.reset_backend()


    # run parallel EMCEE sampler (reference) and time it
    moves_emcee = [
        (emcee.moves.StretchMove(), 0.8),
        (emcee.moves.DESnookerMove(), 0.2),
    ]
    backend_emcee = emcee.backends.HDFBackend(f'Affine_backend_test.h5') 
    n_walkers_emcee = n_walkers*n_ensembles
    n_steps_emcee = n_steps*n_mixing_steps + n_final_steps
    p_0_emcee = np.random.rand(n_walkers_emcee, n_dim)
    t0 = time.time()
    with mp.Pool(n_cores) as pool:
        sampler_emcee = emcee.EnsembleSampler(n_walkers_emcee, n_dim, log_prob, args=log_prob_args, pool=pool, backend=backend_emcee, moves=moves_emcee)
        sampler_emcee.run_mcmc(p_0_emcee,n_steps_emcee, thin=thin)
    samples_emcee_p = sampler_emcee.get_chain(flat=True)
    print(np.shape(samples_emcee_p))
    t = time.time() - t0
    n_log_prob_calls = n_walkers_emcee*n_steps_emcee
    print(f'{n_cores} core(s), affine: {t} sec wall-clock for {n_log_prob_calls} log probability calculations --> {n_log_prob_calls/t} calculations/sec')

    # plot 1d marginal posteriors
    fig, axs = plt.subplots(n_dim, figsize=(10,10))
    plt.suptitle('1d marginal posterior distributions')
    for i, ax in enumerate(axs.flatten()):
        ax.hist(np.transpose(samples)[i], 100, color="blue", histtype="step", density=True, label='parallel w/ mixing')
        ax.hist(np.transpose(samples_2)[i], 100, color="green", histtype="step", density=True, label='parallel w/o mixing')
        ax.hist(np.transpose(samples_emcee_p)[i], 100, color="red", histtype="step", density=True, label='parallel EMCEE')
        ax.legend()
    plt.savefig('parallel_affine_example_dist.png')


    ### example of manually doing mixing and saving samples to .csv
    sampler = pa.ParallelEnsembleSampler(n_ensembles, n_walkers, n_dim, log_prob, log_prob_args, thin, backend_fnames, moves)
    p_0_tmp = p_0
    for i in range(n_mixing_steps):
        states = sampler.run_sampler(p_0, n_steps, n_cores, thin) 
        samples = sampler.get_flat_samples()
        np.savetxt(f"samples_s{i}.csv", samples, delimiter=",")
        p_0_tmp = sampler.mix_ensembles()
        sampler.reset_backend()
    states = sampler.run_sampler(p_0, n_final_steps, n_cores, thin) 
    samples = sampler.get_flat_samples()
    np.savetxt("samples_s_final.csv", samples, delimiter=",")
