import parallel_affine as pa
import parallel_affine_utility as pau
import numpy as np
import time
import emcee 
import multiprocessing as mp
import matplotlib.pyplot as plt
import h5py


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
    n_walkers = 10  # n walkers per ensemble --> (n ensembles * n walkers) total walkers 
    n_mixing_stages = 10
    n_steps_list = [1000 for _ in range(n_walkers)]  # number of steps for each mixing step
    #n_burn_in = 100
    n_total_samples = np.sum(np.array(n_steps_list)*n_walkers*n_ensembles*n_mixing_stages)
    
    log_prob_args = [means, cov]
    n_cores = n_ensembles
    thin = 10  # keep 1/10th of the samples

    # initial start points w/ shape = (n_ensembles, n_walkers, n_dim)
    p_0 = np.array([np.random.rand(n_walkers, n_dim) for i in range(n_ensembles)])

    # set backend filenames
    backend_fnames = [f'PAIES_example_mixing_{i}.h5' for i in range(n_ensembles)]

    # set move set
    moves = [[
        (emcee.moves.StretchMove(), 0.8),
        (emcee.moves.DESnookerMove(), 0.2),
    ] for i in range(n_ensembles)]

    # set run ids
    run_id_list = [ f'run_{i}' for i in range(n_mixing_stages)]

    # run parallel sampler (w/ mixing) and time it
    t0 = time.time()
    sampler = pa.ParallelEnsembleSampler(n_ensembles, n_walkers, n_dim, log_prob, log_prob_args, thin, backend_fnames, moves)
    states = sampler.run_mixing_sampler(p_0, n_steps_list, n_cores, n_mixing_stages,thin,run_id_list)
    t = time.time()-t0
    print(f"wall clock time: {n_total_samples} total samples in {t} s -->  {n_total_samples/t} samples/s")
    pau.plot_single_ensemble_mixing_distributions(sampler)  # quick plot distributions for each ensemble
    D = pau.get_data_all_runs(sampler, flat=True)
    print(f'shape of flattened data (n ensembles, n mixing stages, n_steps/thin, n_dim): {np.shape(D)}')

