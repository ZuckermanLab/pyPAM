import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import emcee 
import sys 
import os
import time
import multiprocessing as mp


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from pyPAM import parallel_affine_utility as pau
from pyPAM import parallel_affine as pa
import toy_model




if __name__ == "__main__": 
    t_pyPAM = []
    t_emcee = []
    n_cpus_list = []

    # use 2-20 number of CPUs 
    for n_cpus_i in range(2,21):
        n_cpus_list.append(n_cpus_i)

        ### parallel affine invariant ensemble sampler with mixing stages
        # sampling parameters 
        total_n_walkers = int(6e3)
        np.random.seed(0)
        n_dim = 2
        n_ensembles = n_cpus_i
        n_walkers = int(total_n_walkers/n_ensembles) # n walkers per ensemble --> (n ensembles * n walkers) total walkers 
        n_mixing_stages = 10
        n_steps_list = [1000 for _ in range(n_mixing_stages)]  # number of steps for each mixing stage
        n_total_samples = np.sum(np.array(n_steps_list))*n_walkers*n_ensembles  # total number of samples collected
        burn_in = int(n_total_samples/4)

        log_prob_args = [1,100]  # additional arguments for log probability function f(x,*args)
        n_cores = n_ensembles  

        # initial start points w/ shape = (n_ensembles, n_walkers, n_dim)
        x0_range = np.random.uniform(-2, 2, size=(n_ensembles, n_walkers, 1))
        x1_range = np.random.uniform(-1, 3, size=(n_ensembles, n_walkers, 1))
        p_0 = np.concatenate((x0_range, x1_range), axis=2)

        # set backend filenames for each ensemble
        backend_fnames = [f'example_data_{i}.h5' for i in range(n_ensembles)]

        # set move set for each ensemble
        moves = [[
            (emcee.moves.StretchMove(), 0.8),
            (emcee.moves.DESnookerMove(), 0.2),
        ] for i in range(n_ensembles)]

        # set run labels for each mixing stage
        run_id_list = [ f'stage_{i}' for i in range(n_mixing_stages)]

        # run parallel sampler (w/ mixing) and time it
        t0 = time.time()
        sampler = pa.ParallelEnsembleSampler(n_ensembles, n_walkers, n_dim, toy_model.log_prob, log_prob_args,  backend_fnames, moves)
        states = sampler.run_mixing_sampler(p_0, n_steps_list, n_cores, n_mixing_stages, run_id_list)
        t = time.time()-t0 
        # Clean up test files (removes h5 files)
        for fname in backend_fnames:
            if os.path.exists(fname):
                os.remove(fname)
        t_pyPAM.append(t)

        ### emcee
        n_walkers_emcee = int(total_n_walkers)#n_walkers*n_ensembles
        n_steps_emcee = np.sum(np.array(n_steps_list))
        total_samples_emcee = n_walkers_emcee*n_steps_emcee
        backend_emcee = emcee.backends.HDFBackend('example_data_emcee.h5', name='init_empty')
        # initial start points w/ shape = (n_ensembles, n_walkers, n_dim)
        x0_range_emcee = np.random.uniform(-2, 2, size=(n_walkers_emcee, 1))
        x1_range_emcee = np.random.uniform(-1, 3, size=(n_walkers_emcee, 1))
        p_0_emcee = np.concatenate((x0_range_emcee, x1_range_emcee), axis=1)


        t0 = time.time()
        with mp.Pool(n_cores) as pool:
            sampler2 = emcee.EnsembleSampler(n_walkers_emcee, n_dim, toy_model.log_prob, args=log_prob_args, backend=backend_emcee, moves=moves[0])
            state2 = sampler2.run_mcmc(p_0_emcee,n_steps_emcee)
        t = time.time()-t0 


        if os.path.exists('example_data_emcee.h5'):
            os.remove('example_data_emcee.h5')

        t_emcee.append(t)
        print(f'{n_cpus_i} cpus: pyPAM={t_pyPAM[-1]}s, emcee={t_emcee[-1]}s, ratio={t_pyPAM[-1]/t_emcee[-1]}s')

   