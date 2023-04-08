r'''
pyPAM is a Parallelized Affine invariant ensemble sampler with an optional Mixing step. It is an extension of the [emcee](https://emcee.readthedocs.io/en/stable/) sampling package.

This method parallelizes multiple ensembles for better performance, and also includes an optional phase of shuffling walkers between ensembles. 


### Getting started

Clone repository:
```bash
git clone https://github.com/ZuckermanLab/pyPAM

```
Example usage:
```python
import parallel_affine as pa
sampler = pa.ParallelEnsembleSampler(n_ensembles, n_walkers, n_dim, log_prob, log_prob_args, thin, backend_fnames, moves)
states = sampler.run_mixing_sampler(p_0, n_steps_list, n_cores, n_mixing_stages,thin,run_id_list)
pa.plot_single_ensemble_mixing_distributions(sampler)  # quick plot distributions for each ensemble
D = pau.get_data_all_runs(sampler, flat=True)  # get data --> shape = (n ensembles, n mixing stages, n_steps/thin, n_dim)
```


'''
