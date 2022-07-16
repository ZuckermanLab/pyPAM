# parallel_affine_invariant_ensemble_sampler
##### August George, 2022, under development
A parallelizable extension of the affine invariant ensemble sampler (EMCEE python package) w/ optional 'mixing' for initialization. Currently supports multiple cores (not multiple HPC nodes yet)


##### Updates

7/13/2022 - now data is stored for all mixing steps, added some utility functions, and updated example script.

7/6/2022 - added 'backends' and 'moves' option, with example usage

7/1/2022 - added 'thin' option, example usage, and example on doing 'manual' mixing affine while saving samples at every stage

---

There are two main capabilities:
1. run several affine invariant ensemble samplers together in parallel
2. run several affine invariant ensemble samplers together in parallel w/ a 'mixing' intialization phase. Here after n steps, the last walkers are shuffled between all the enesembles and used as the intial walker positions for the next sampling iteration. This is repeated m times until a final 'production' run is started. 


Contents:
1. main code (parallel_affine.py), 
2. example (parallel_affine_example.py), 
3. slurm scheduler script (run_sampler_slurm.sh), 
4. sge scheduler script (run_sampler_sge.sh - not finished yet!)
    + need to convert from slurm to sge: https://srcc.stanford.edu/sge-slurm-conversion  

Quick start (see example and code comments for more details):
1. running several affine invariant ensemble samplers together in parallel
```
import parallel_affine as pa
sampler = pa.ParallelEnsembleSampler(n_ensembles, n_walkers, n_dim, log_prob, log_prob_args, thin)
states = sampler.run_sampler(p_0, n_steps_total, n_cores, thin)
samples = sampler.get_flat_samples()  # shape = (n_ensembles*n_steps*n_walkers,n_dim)
```
2. running several affine invariant ensemble samplers together in parallel, w/ mixing initialization
```
import parallel_affine as pa
sampler = pa.ParallelEnsembleSampler(n_ensembles, n_walkers, n_dim, log_prob, log_prob_args, thin)
states = sampler.run_mixing_sampler(p_0, n_steps, n_cores, n_mixing_steps, n_final_steps, thin)  
samples = sampler.get_flat_samples()  # shape = (n_ensembles*n_final_steps*n_walkers,n_dim)
```

Running on HPC:
1. install a python environment and clone github project on HPC 
2. modify scheduling scripts based on your needs/environment
3. run example to ensure it's working as expected




