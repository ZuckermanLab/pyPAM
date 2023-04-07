# pyPAM
A parallel extension of the affine invariant ensemble sampler (emcee) w/ optional ensemble mixing


Clone repository:
```
git clone https://github.com/ZuckermanLab/pyPAM
```
Usage:
```
import parallel_affine as pa
sampler = pa.ParallelEnsembleSampler(n_ensembles, n_walkers, n_dim, log_prob, log_prob_args, thin, backend_fnames, moves)
states = sampler.run_mixing_sampler(p_0, n_steps_list, n_cores, n_mixing_stages,thin,run_id_list)
pa.plot_single_ensemble_mixing_distributions(sampler)  # quick plot distributions for each ensemble
D = pau.get_data_all_runs(sampler, flat=True)  # get data --> shape = (n ensembles, n mixing stages, n_steps/thin, n_dim)
```

August George, Zuxkerman Lab, OHSU, 2023
