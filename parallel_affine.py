# August George, 2022. Runs class and wrapper function to run emcee (affine sampler) in parallel

import emcee 
import numpy as np
import multiprocessing as mp
import types
import sys
import h5py


def wrapper(arg_list):
    """wrapper function to make and run ensemble sampler object in parallel"""
    n_walkers = arg_list[0]
    n_dim = arg_list[1]
    log_prob = arg_list[2]
    log_prob_args = arg_list[3]
    p_0 = arg_list[4]
    n_steps = arg_list[5]
    thin = arg_list[6]
    backend = arg_list[7]
    moves = arg_list[8]
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob, args=log_prob_args, backend=backend, moves=moves)
    state = sampler.run_mcmc(p_0,n_steps, thin=thin)
    return (sampler, state, backend)


class ParallelEnsembleSampler:
    def __init__(self, n_ensembles, n_walkers, n_dim, log_prob, log_prob_args, thin, backend_fnames, moves):
        """initializes sampler class"""
        self.n_ensembles = n_ensembles
        self.n_walkers = n_walkers
        self.n_dim = n_dim
        self.log_prob = log_prob
        self.log_prob_args = log_prob_args
        self.thin = thin
        self.backend_fnames = backend_fnames
        self.moves_list = moves
        # check that inputs are valid
        assert(type(self.n_ensembles)==int and self.n_ensembles>=1)
        assert(type(self.n_dim)==int and self.n_dim>=1)
        assert(type(self.n_walkers)==int and self.n_walkers>=2*n_dim and self.n_walkers%2==0)
        assert(isinstance(self.log_prob, types.FunctionType)==True)
        assert(type(self.log_prob_args)==list)
        assert(type(self.thin)==int and self.thin>=1)
        assert(type(self.backend_fnames)==list and len(self.backend_fnames)==self.n_ensembles)
        assert(type(self.moves_list)==list and len(self.moves_list)==self.n_ensembles)
  
          

        #assert(type(self.moves)==list)

        # Initialize the sampler
        # create a list ('ensemble') of backends and affine invariant ensemble samplers
        try:
            self.backend_list = [emcee.backends.HDFBackend(self.backend_fnames[i], name=f"ensemble_{i}") for i in range(self.n_ensembles)] 
            assert(all(type(item) is emcee.backends.HDFBackend for item in self.backend_list)==True)  # check that list was created correctly
            ### use backend.reset()?
            self.sampler_list = [emcee.EnsembleSampler(self.n_walkers, self.n_dim, self.log_prob, args=self.log_prob_args, backend=self.backend_list[i], moves=self.moves_list[i]) for i in range(self.n_ensembles)]
            assert(all(type(item) is emcee.EnsembleSampler for item in self.sampler_list)==True)  # check that list was created correctly
        except:
            print("Error: could not create a list of affine invariant ensemble samplers", file=sys.stderr)
            sys.exit(1)

     
   
    def run_sampler(self,p_0,n_steps,n_cores,thin):
        """runs parallel ensemble samplers - without mixing"""
        # note: self.n_steps gets updated during self.run_sampler() call
        self.p_0 = p_0
        self.n_steps = n_steps
        self.n_cores = n_cores
        self.thin = thin

        # check that inputs are valid        
        assert(isinstance(self.p_0,(list,np.ndarray)) and np.shape(self.p_0) == (self.n_ensembles, self.n_walkers, self.n_dim))
        assert(type(self.n_steps)==int and self.n_steps>=1)
        assert(type(self.n_cores)==int and self.n_cores>=1)
        #assert(type(self.moves)==list)
        assert(type(self.thin)==int and self.thin>=1)

        # create a list of all the arguments for 'wrapper' function that makes and runs affine samplers in parallel
        arg_list = [(self.n_walkers, self.n_dim, self.log_prob, self.log_prob_args, self.p_0[i], self.n_steps, self.thin, self.backend_list[i], self.moves_list[i]) for i in range(self.n_ensembles)]
        pool = mp.Pool(self.n_cores)  # create a pool of workers w/ n cores
        r = pool.map(wrapper, arg_list)  # use pool w/ map to run affine samplers in parallel
        pool.close()   

        # process results into a list of samplers and sampler states
        new_sampler_list = [ i[0] for i in r]
        state_list = [i[1] for i in r]
        new_backend_list = [i[2] for i in r]
        self.sampler_list = new_sampler_list  # update self w/ latest sampling results
        self.backend_list = new_backend_list # update self w/ latest backend results
        return state_list  


    def get_chains(self):
        """returns the MCMC chains (samples) w/ shape = (n_ensembles, n_steps, n_walkers, n_dim)"""
        samples = [i.get_chain() for i in self.sampler_list]
        assert(np.shape(samples)==(self.n_ensembles,int(self.n_steps/self.thin),self.n_walkers,self.n_dim))
        return np.array(samples)


    def get_last_samples(self):
        """returns the last walker positions (last sample points) for each ensemble w/ shape = (n_ensembles, n_walkers, n_dim) """
        samples = self.get_chains()  # shape = (n ensembles, n steps/walker, n walkers, n dim)
        last_samples = samples[:,-1]
        # check that last samples are correct. - use alternative way to get last samples using emcee sampler 
        last_samples_emcee = np.array([i.get_last_sample()[0] for i in self.sampler_list])
        assert(np.shape(last_samples)==np.shape(last_samples_emcee))
        assert(np.array_equal(last_samples,last_samples_emcee))  # might be slow for large arrays
        return last_samples


    def mix_ensembles(self):
        """shuffle last examples to use as initial walker positions for next sampling run w/ shape = (n_ensembles, n_walkers, n_dim)"""
        last_samples = self.get_last_samples()
        # combine last samples from the ensemple sampler, so each row is a walker coordinate 
        # old shape (n_ensembles, n_walkers, n_dim) --> new shape (n_ensembles*n_walkers, n_dim)
        last_samples_agg = np.reshape(last_samples, (self.n_ensembles*self.n_walkers, self.n_dim))
        np.random.shuffle(last_samples_agg)  # shuffle each row (each walker)
        p0_new = np.reshape(last_samples_agg, (self.n_ensembles, self.n_walkers, self.n_dim))  # return to original shape
        return p0_new  


    def run_mixing_sampler(self,p_0,n_steps,n_cores,n_mixing_steps,n_final_steps,thin):
        """runs parallel ensemble samplers - with N mixing steps plus a final sampling run"""
        # check that inputs are valid        
        assert(isinstance(p_0,(list,np.ndarray)) and np.shape(p_0) == (self.n_ensembles, self.n_walkers, self.n_dim))
        assert(type(n_steps)==int and n_steps>=1)
        assert(type(n_cores)==int and n_cores>=1)
        assert(type(n_mixing_steps)==int and n_mixing_steps>=0)
        assert(type(n_final_steps)==int and n_final_steps>=n_steps)
        #assert(type(moves)==list)
        assert(type(thin)==int and thin>=1)


        p_0_tmp = p_0  # start w/ initial parameter set 
        for i in range(n_mixing_steps):
            # start parallel affine invariant ensemble samplers
            self.run_sampler(p_0_tmp,n_steps,n_cores,thin)
            p_0_tmp = self.mix_ensembles()  # update starting parameter set to new 'shuffled' parameter sets
            self.reset_backend()
        # note: self.n_steps gets updated during self.run_sampler() call, so will update to n_final_steps
        state_list = self.run_sampler(p_0_tmp,n_final_steps,n_cores,thin)     
        return state_list


    def get_flat_samples(self):
        """return samples in a flat shape - (n_ensembles*n_steps*n_walkers, n_dim)"""
        chains = self.get_chains()
        flat_samples = np.reshape(chains, (self.n_ensembles*int(self.n_steps/self.thin)*self.n_walkers, self.n_dim))
        return flat_samples


    def print_backend_sampling_summary(self):
        """print the shape of the samples loaded from the backend """
        for i, reader in enumerate(self.backend_list):
            samples = reader.get_chain(flat=True, thin=self.thin)
            print(f'backend: {reader}')
            print(f'fname: {self.backend_fnames[i]}')
            print(f'  flat samples shape: {np.shape(samples)}')
    

    def reset_backend(self):
        """resets backend - removing previous samples"""
        for b in self.backend_list:
            b.reset(self.n_walkers, self.n_dim)
            print('warning: resetting backend (removing old samples from chain)')

        


    
if __name__ == "__main__": 
    pass
   

