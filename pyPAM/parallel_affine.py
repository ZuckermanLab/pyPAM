# August George, 2023

import emcee 
import numpy as np
import multiprocessing as mp
import types
import sys
import pyPAM.parallel_affine_utility as pau


class ParallelEnsembleSampler:
    """Parallel affine invariant ensemble sampler for MCMC.

    This class implements the parallel affine invariant ensemble sampler algorithm for Markov Chain Monte Carlo (MCMC) simulations. It is designed to be run on multiple processors or cores, and is intended for use with high-dimensional problems that are difficult to sample from using traditional MCMC methods.

    Args:
        n_ensembles (int): The number of independent ensembles to run.
        n_walkers (int): The total number of walkers across all ensembles. Should be an even multiple of n_dim.
        n_dim (int): The number of dimensions in the problem space.
        log_prob (function): A function that computes the log probability of a given point in the problem space.
        log_prob_args (list): A list of additional arguments to pass to log_prob.
        backend_fnames (list): A list of file names for the backends to use. There should be one filename for each ensemble.
        moves (list): A list of moves to use for each ensemble.

    Attributes:
        n_ensembles (int): The number of independent ensembles to run.
        n_walkers (int): The total number of walkers across all ensembles.
        n_dim (int): The number of dimensions in the problem space.
        log_prob (function): A function that computes the log probability of a given point in the problem space.
        log_prob_args (list): A list of additional arguments to pass to log_prob.
        backend_fnames (list): A list of file names for the backends to use.
        moves_list (list): A list of moves to use for each ensemble.
        backend_list (list): A list of backends for each ensemble.
        sampler_list (list): A list of ensemble samplers for each ensemble.

    Raises:
        AssertionError: If any of the input arguments are invalid.

    """

    def __init__(self, n_ensembles, n_walkers, n_dim, log_prob, log_prob_args, backend_fnames, moves):
        """initializes sampler class"""
        self.n_ensembles = n_ensembles
        self.n_walkers = n_walkers
        self.n_dim = n_dim
        self.log_prob = log_prob
        self.log_prob_args = log_prob_args
        self.backend_fnames = backend_fnames
        self.moves_list = moves
      
        # check that inputs are valid
        assert(type(self.n_ensembles)==int and self.n_ensembles>=1)
        assert(type(self.n_dim)==int and self.n_dim>=1)
        assert(type(self.n_walkers)==int and self.n_walkers>=2*n_dim and self.n_walkers%2==0)
        assert(isinstance(self.log_prob, types.FunctionType)==True)
        assert(type(self.log_prob_args)==list)
        assert(type(self.backend_fnames)==list and len(self.backend_fnames)==self.n_ensembles)
        assert(type(self.moves_list)==list and len(self.moves_list)==self.n_ensembles)

        # Initialize the sampler
        # create a list ('ensemble') of backends and affine invariant ensemble samplers
        try:
            self.backend_list = [emcee.backends.HDFBackend(self.backend_fnames[i], name='init_empty') for i in range(self.n_ensembles)] 
            assert(all(type(item) is emcee.backends.HDFBackend for item in self.backend_list)==True)  # check that list was created correctly
            self.sampler_list = [emcee.EnsembleSampler(self.n_walkers, self.n_dim, self.log_prob, args=self.log_prob_args, backend=self.backend_list[i], moves=self.moves_list[i]) for i in range(self.n_ensembles)]
            assert(all(type(item) is emcee.EnsembleSampler for item in self.sampler_list)==True)  # check that list was created correctly
        except:
            print("Error: could not create a list of affine invariant ensemble samplers", file=sys.stderr)
            sys.exit(1)

    
    def update_backends(self, id):
        """Update the backends to include a new section.

        This method updates the HDF5 backend for each ensemble by adding a new section to the file. The new section is identified by an ID string, which is appended to the name of the section.

        Args:
            id (str): The ID string to use for the new section.

        Raises:
            AssertionError: If the input ID is not a string, or is an empty string.
        """
        self.id = id
        assert(type(self.id)==str and len(self.id)>=1)
        self.backend_list = [emcee.backends.HDFBackend(self.backend_fnames[i], name=f"{self.id}_ensemble_{i}") for i in range(self.n_ensembles)] 
        assert(all(type(item) is emcee.backends.HDFBackend for item in self.backend_list)==True)  # check that list was created correctly

   
    def run_sampler(self, p_0, n_steps, n_cores, id):
        """Run the parallel ensemble samplers.

        This method runs the parallel affine invariant ensemble samplers for a specified number of steps, with a given initial state and number of cores. It updates the backends for each ensemble to include a new section identified by an ID string, and returns the final state of the samplers.

        Args:
            p_0 (ndarray): The initial state for each walker in each ensemble, as a 3D array with shape (n_ensembles, n_walkers, n_dim).
            n_steps (int): The number of steps to run each sampler for.
            n_cores (int): The number of CPU cores to use for parallelization.
            id (str): The ID string to use for the new backend section.

        Returns:
            state_list (list): A list of sampler states, one for each ensemble.

        Raises:
            AssertionError: If any of the input arguments are invalid.
        """
        # note: self.n_steps gets updated during self.run_sampler() call
        self.p_0 = p_0
        self.n_steps = n_steps
        self.n_cores = n_cores

        # check that inputs are valid        
        assert(isinstance(self.p_0,(list,np.ndarray)) and np.shape(self.p_0) == (self.n_ensembles, self.n_walkers, self.n_dim))
        assert(type(self.n_steps)==int and self.n_steps>=1)
        assert(type(self.n_cores)==int and self.n_cores>=1)
        assert(type(id)==str and len(id)>=1)  # redundant?!

        # update backends for new id
        self.update_backends(id)

        # create a list of all the arguments for 'wrapper' function that makes and runs affine samplers in parallel
        arg_list = [(self.n_walkers, self.n_dim, self.log_prob, self.log_prob_args, self.p_0[i], self.n_steps,  self.backend_list[i], self.moves_list[i]) for i in range(self.n_ensembles)]
        with mp.Pool(self.n_cores) as pool:  # create a pool of workers w/ n cores
            r = pool.map(pau.wrapper, arg_list)  # use pool w/ map to run affine samplers in parallel

        # process results into a list of samplers and sampler states
        new_sampler_list = [ i[0] for i in r]
        state_list = [i[1] for i in r]
        new_backend_list = [i[2] for i in r]
        self.sampler_list = new_sampler_list  # update self w/ latest sampling results
        self.backend_list = new_backend_list # update self w/ latest backend results
        return state_list  
    

    def get_chains(self):
        """Return the MCMC chains (samples) for each ensemble.

        This method returns the MCMC chains (samples) for each ensemble as a 4D numpy array with shape (n_ensembles, n_steps, n_walkers, n_dim).

        Returns:
            samples (ndarray): The MCMC chains for each ensemble.

        Raises:
            AssertionError: If the shape of the returned samples is not as expected.
        """
        samples = [i.get_chain() for i in self.sampler_list]
        actual_shape = np.shape(samples)
        expected_shape = (self.n_ensembles, int(self.n_steps), self.n_walkers, self.n_dim)
        assert(np.shape(samples)==(self.n_ensembles,int(self.n_steps),self.n_walkers,self.n_dim))
        return np.array(samples)


    def get_last_samples(self):
        """Return the last walker positions for each ensemble.

        This method returns the last walker positions (last sample points) for each ensemble as a 3D numpy array with shape (n_ensembles, n_walkers, n_dim).

        Returns:
            last_samples (ndarray): The last walker positions for each ensemble.

        Raises:
            AssertionError: If the last samples returned by the method are not correct.
        """
        samples = self.get_chains()  # shape = (n ensembles, n steps/walker, n walkers, n dim)
        last_samples = samples[:,-1]
        # check that last samples are correct. - use alternative way to get last samples using emcee sampler 
        last_samples_emcee = np.array([i.get_last_sample()[0] for i in self.sampler_list])
        assert(np.shape(last_samples)==np.shape(last_samples_emcee))
        np.testing.assert_allclose(last_samples,last_samples_emcee)  # can use array_equal for slower but more accurate method
        return last_samples


    def mix_ensembles(self):
        """Shuffle the last samples to use as initial walker positions for the next sampling run.

        This method shuffles the last samples from each ensemble sampler and combines them into a new set of initial walker positions for the next sampling run.

        Returns:
            p0_new (ndarray): The new set of initial walker positions, as a 3D numpy array with shape (n_ensembles, n_walkers, n_dim).
        """

        last_samples = self.get_last_samples()
        # combine last samples from the ensemple sampler, so each row is a walker coordinate 
        # old shape (n_ensembles, n_walkers, n_dim) --> new shape (n_ensembles*n_walkers, n_dim)
        last_samples_agg = np.reshape(last_samples, (self.n_ensembles*self.n_walkers, self.n_dim))
        np.random.shuffle(last_samples_agg)  # shuffle each row (each walker)
        p0_new = np.reshape(last_samples_agg, (self.n_ensembles, self.n_walkers, self.n_dim))  # return to original shape
        return p0_new  


    def run_mixing_sampler(self, p_0, n_steps_list, n_cores, n_mixing_stages, id_list):
        """Run the parallel ensemble samplers with N mixing steps plus a final sampling run.

        This method runs the parallel affine invariant ensemble samplers with a specified number of mixing steps, followed by a final sampling run. It takes an initial state, a list of numbers of steps to run at each mixing stage, the number of CPU cores to use for parallelization, the number of mixing stages to run, and a list of ID strings for the backend sections for each mixing stage.

        Args:
            p_0 (ndarray): The initial state for each walker in each ensemble, as a 3D array with shape (n_ensembles, n_walkers, n_dim).
            n_steps_list (list): A list of integers specifying the number of steps to run each sampler for at each mixing stage.
            n_cores (int): The number of CPU cores to use for parallelization.
            n_mixing_stages (int): The number of mixing stages to run.
            id_list (list): A list of ID strings for the backend sections for each mixing stage.

        Returns:
            state_list (list): The final state of the samplers, as a list of sampler states.

        Raises:
            AssertionError: If any of the input arguments are invalid.
        """
        # check that inputs are valid        
        assert(isinstance(p_0,(list,np.ndarray)) and np.shape(p_0) == (self.n_ensembles, self.n_walkers, self.n_dim))
        assert(type(n_steps_list)==list and all((type(item) is int) and (item > 0) for item in n_steps_list)==True)
        assert(type(n_cores)==int and n_cores>=1)
        assert(type(n_mixing_stages)==int and n_mixing_stages>=0)   
        assert(type(id_list)==list and len(id_list)==n_mixing_stages and (all(type(item) is str for item in id_list)==True))
        self.id_list = id_list

        p_0_tmp = p_0  # start w/ initial parameter set 
        for i in range(n_mixing_stages):
            tmp_id = id_list[i]
            tmp_n_steps = n_steps_list[i]
            # start parallel affine invariant ensemble samplers
            state_list = self.run_sampler(p_0_tmp,tmp_n_steps,n_cores,tmp_id)
            p_0_tmp = self.mix_ensembles()  # update starting parameter set to new 'shuffled' parameter sets
        # note: self.n_steps gets updated during self.run_sampler() call, so will update to n_final_steps    
        return state_list  # state list of final iteration


    def get_flat_samples(self):
        """Return the MCMC samples in a flat shape.

        This method returns the MCMC samples in a flat shape, with dimensions (n_ensembles*n_steps*n_walkers, n_dim).

        Returns:
            flat_samples (ndarray): The flattened MCMC samples.

        """
        chains = self.get_chains()
        flat_samples = np.reshape(chains, (self.n_ensembles*int(self.n_steps)*self.n_walkers, self.n_dim))
        return flat_samples


    def reset_backends(self):
        """Reset the backends, removing all previous samples.

        This method resets the backends by removing all previous samples from the MCMC chains. This can be useful when starting a new sampling run from scratch.

        Raises:
            Warning: A warning message is printed to the console indicating that the backend has been reset.
        """
        for b in self.backend_list:
            b.reset(self.n_walkers, self.n_dim)
            print('warning: resetting backend (removing old samples from chain)')

        

if __name__ == "__main__": 
    pass
   

