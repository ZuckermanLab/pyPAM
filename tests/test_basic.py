import unittest
import numpy as np
from numpy.testing import assert_array_equal
import emcee
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0,  os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyPAM import parallel_affine as pa
from pyPAM import parallel_affine_utility as pau


    
def log_prob(x):
    return -0.5*np.sum(x**2)

class TestParallelEnsembleSampler(unittest.TestCase):
    def test_run_sampler(self):
        # Define test inputs
        n_ensembles = 2
        n_walkers = 10
        n_dim = 3
        cov = 0.1 * np.eye(n_dim) 
        backend_fnames = ['./tests/test_backend_0.h5', './tests/test_backend_1.h5']
        moves = [emcee.moves.GaussianMove(cov) for _ in range(n_ensembles)]
        p_0 = np.zeros((n_ensembles, n_walkers, n_dim))
        p_0 += np.random.normal(scale=1e-4, size=p_0.shape)
        n_steps = 10
        n_cores = 2
        id = 'test_section'

        # Initialize sampler object
        sampler = pa.ParallelEnsembleSampler(n_ensembles, n_walkers, n_dim, log_prob, [],  backend_fnames, moves)

        # Run sampler
        state_list = sampler.run_sampler(p_0, n_steps, n_cores,  id)

        # Test output types and shapes
        self.assertIsInstance(state_list, list)
        self.assertEqual(len(state_list), n_ensembles)
        for i in range(n_ensembles):
            self.assertIsInstance(state_list[i], emcee.state.State)

        # Test that initial positions are updated correctly
        new_p_0 = np.zeros((n_ensembles, n_walkers, n_dim))
        new_p_0 += np.random.normal(scale=1e-4, size=new_p_0.shape)
        state_list = sampler.run_sampler(new_p_0, n_steps, n_cores, id)
        assert_array_equal(sampler.p_0, new_p_0)

        # Test that the sampler raises an error if the initial state has a large condition number
        p_0 = np.zeros((n_ensembles, n_walkers, n_dim))
        with self.assertRaises(ValueError):
            state_list = sampler.run_sampler(p_0, n_steps, n_cores, id)

        # Test that the sampler raises an error if the input arguments are invalid
        with self.assertRaises(AssertionError):
            state_list = sampler.run_sampler(p_0, -1, n_cores, id)

        # Clean up test files
        for fname in backend_fnames:
            if os.path.exists(fname):
                os.remove(fname)




if __name__ == '__main__':
    unittest.main()