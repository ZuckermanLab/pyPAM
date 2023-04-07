# August George, 2022. Runs class and wrapper function to run emcee (affine sampler) in parallel

import emcee 
import numpy as np
import h5py
import matplotlib.pyplot as plt


def plot_single_ensemble_mixing_distributions(paie_sampler, fname='parallel_affine_example_dist', n_bins=100, xlim=[[-1,1]]):
    """creates seperate distribution figures for each ensemble - using data from each parameter and mixing stage """
    assert(type(fname)==str and len(fname)>=1)
    assert(type(n_bins)==int and n_bins>1)
    assert(type(xlim)==list and xlim[0][1]>xlim[0][0])
    s = paie_sampler
    data_list = get_data_all_runs(s, flat=True)
    n_dim = s.n_dim
    for k, ensemble_data in enumerate(data_list):
        # plot 1d marginal posteriors
        fig, axs = plt.subplots(n_dim, figsize=(15,12))
        plt.suptitle('1D marginal posterior distributions - ensemble {k}')
        if n_dim >=2:
            for i, ax in enumerate(axs.flatten()):  # for each subplot figure (parameter)
                for j, D_j in enumerate(ensemble_data):  # for each mixing stage
                    D_tmp = np.transpose(D_j)
                    ax.hist(D_tmp[i], n_bins, histtype="step", density=True, label=f'idx {j}', range=(xlim[i][0], xlim[i][1]))   # plot parameter histogram
                    ax.legend()
                ax.set_title(f'p_{i} distribution')
                ax.set_xlim(xlim[0][0], xlim[0][1])

        else:
            ax = axs
            i=0
            for j, D_j in enumerate(ensemble_data):  # for each mixing stage
                D_tmp = np.transpose(D_j)
                ax.hist(D_tmp[i], n_bins, histtype="step", density=True, label=f'idx {j}',range=(xlim[i][0], xlim[i][1]))   # plot parameter histogram
                ax.legend()
                ax.set_title(f'p_{j} distribution')
            ax.set_xlim(xlim[i][0], xlim[i][1])

        plt.tight_layout()
        plt.savefig(f'{fname}_{k}.png')
        plt.close()
    return fig, axs



def get_data_all_runs(paie_sampler, flat=False):
    """gets data from multiple runs and multiple ensembles. Outputs a list of datasets. D[0] = data from all runs from ensemble 0. D[0][0] = data from run 0 of ensemble 0"""
    s = paie_sampler
    backend_fnames = s.backend_fnames
    data_list = []
    # load every datafile
    for bfname in backend_fnames:
        ensemble_list = []
        f = h5py.File(bfname, 'r')
        for key in f.keys():
            if key != 'init_empty':
                c_tmp = f[key]['chain']
                if flat ==True:
                    c = np.reshape(c_tmp, (c_tmp.shape[0]*c_tmp.shape[1],c_tmp.shape[2]))
                else:
                    c = c_tmp
                ensemble_list.append(c)
            else:
                pass
        data_list.append(ensemble_list)

    return data_list


def get_data_dict_from_backend(fname, flat=False):
    assert(type(fname)==str and len(fname)>=1)
    f = h5py.File(fname,'r')
    ensemble_data_dict = {}
    for key in f.keys():
        if key != 'init_empty':
            c_tmp = f[key]['chain']
            print(np.shape(c_tmp))
            if flat ==True:
                c = np.reshape(c_tmp, (c_tmp.shape[0]*c_tmp.shape[1],c_tmp.shape[2])) 
            else:
                c = c_tmp
            ensemble_data_dict[key] = c
        else:
            pass
    return ensemble_data_dict




def print_backend_info(fname):
    """prints info on the hdf5 file contents"""
    with h5py.File(fname, "r") as f:
        for key in list(f.keys()):
            print(f'dataset: {key}')
            print(f'dataset sections: {f[key].keys()}')
            d_tmp = f[key]['chain']
            print(f'chain shape: {d_tmp.shape}')


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
    #tmp_samples = sampler.get_chain(flat=True)
    #print_samples_description(tmp_samples)
    return (sampler, state, backend)


if __name__ == "__main__": 
    pass
   

