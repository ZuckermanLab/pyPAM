# parallel_affine_invariant_ensemble_sampler
A parallelizable extension of the affine invariant ensemble sampler (EMCEE python package) w/ optional 'mixing' for initialization. 

There are two main capabilities:
1. run several affine invariant ensemble samplers together in parallel
2. run several affine invariant ensemble samplers together in parallel w/ a 'mixing' intialization phase. Here after n steps, the last walkers are shuffled between all the enesembles and used as the intial walker positions for the next sampling iteration. This is repeated m times until a final 'production' run is started. 

August George, 2022, under development

Contents:
1. main code (parallel_affine.py), 
2. example (parallel_affine_example.py), 
3. slurm scheduler script (run_sampler_slurm.sh), 
4. sge scheduler script (run_sampler_sge.sh - not finished yet)

Quick start:
1. running several affine invariant ensemble samplers together in parallel
```
import parallel_affine as pa
sampler = pa.ParallelEnsembleSampler(n_ensembles, n_walkers, n_dim, log_prob, log_prob_args)
states = sampler.run_sampler(p_0, n_steps_total, n_cores)
samples = sampler.get_flat_samples()  # shape = (n_ensembles*n_steps*n_walkers,n_dim)
```
