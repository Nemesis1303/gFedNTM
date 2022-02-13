# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                           UTILS EVALUATION                             ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
import numpy as np
import pandas as pd
from scipy import sparse


def get_simmat_thetas(from_file, n_docs, thetas=None, path_to_model=None):
    """Gets the similarity matrix of a topic model's document-topic distribution.

    Args:
    -----
        * from_file (boolean):                    True if the thetas matrix is going to be 
                                                  provided through a file.
        * n_docs (int):                           Number of documents with which the model was 
                                                  trained.
        * thetas (darray, optional):              Document-topic distribution of the model 
                                                  whose performance is being evaluated. Defaults to None.
        * path_to_model (pathlib.Path, optional): Path to the model.npz file in which the 
                                                  model is located in case the thetas matrix needs to be read from file. Defaults to None.

    Returns:
    --------
        * ndarry: Documents similarity matrix of the requested model.
    """
    if from_file:
        if path_to_model:
            with np.load(path_to_model) as data:
                thetas = sparse.csr_matrix(
                    (data['thetas_data'], data['thetas_indices'], data['thetas_indptr']), shape=data['thetas_shape'])[:n_docs, :]
            thetas_sqrt = np.sqrt(thetas).toarray()
        else:
            print("No path to model file was provided")
            return None
    else:
        if thetas is not None:
            thetas_sqrt = np.sqrt(thetas)
        else:
            print("No thetas matrix was provided")
            return None
    sim_mat = thetas_sqrt.dot(thetas_sqrt.T)
    return sim_mat


def get_simmat_betas(from_file, betas=None, betas_orig=None, path_to_model=None):
    """Gets the similarity matrix between the ground truth and model inferred word-topic distributions. 

    Args:
    -----
        * from_file (boolean):                    True if the betas matrixes are going to be 
                                                  provided through a file.
        * betas (ndarray, optional):              Topic model's word-topic distribution. 
                                                  Defaults to None.
        * betas_orig (ndaaray, optional):         Ground truth word-topic distribution. 
                                                  Defaults to None.
        * path_to_model (pathlib.Path, optional): Path to the model.npz file in which the 
                                                  model is located in case the betas matrixes need to be read from file. Defaults to None.

    Returns:
    --------
        * ndarray: Betas similarity matrix of the requested model.
    """
    if from_file:
        data = np.load(from_file)
        betas = data['betas']
        betas_orig = data['betas_orig']
    else:
        if betas is None or betas_orig is None:
            print("One of the betas matrixes was not provided.")
            return None
    betas_sqrt = np.sqrt(betas)
    betas_orig_sqrt = np.sqrt(betas_orig)
    sim_mat = betas_sqrt.dot(betas_orig_sqrt.T)
    return sim_mat


def get_average_std_nruns(from_file, n_docs, n_runs, method, base_path=None, thetas=None):
    """Calculates the average standard deviation of the documents similarity matrix obtained from n independent executions of the training of the topic model.

    Args:
    -----
        * from_file (boolean):                True if the thetas matrix is going to be 
                                              provided through a file.
        * n_docs (int):                       Nr of documents of the training data with which 
                                              the model was trained.
        * n_runs (int):                       Nr of times the traininig of the model has been 
                                              carried out. 
        * method (str):                       Method that is going to be used for the       
                                              calculation of the average std:
                                                - normal: average std as it is
                                                - percentage: average std of 10 % maximum values according to mean
        * base_path (pathlib.Path, optional): Path where the "model.npz" files are located. Defaults to None.
        * thetas (darray, optional):          Document-topic distribution of the model 
                                              whose performance is being evaluated. Defaults to None.

    Returns:
    --------
        * float: Standard desviation of the variability
    """
    mean_sim = np.zeros((n_docs, n_docs))
    quad_sim = np.zeros((n_docs, n_docs))
    for run in np.arange(n_runs):
        if from_file:
            if base_path:
                model_name = 'model_run_' + str(run)
                path_model = \
                    base_path.joinpath(model_name).joinpath('modelo.npz')
                sim_mat = \
                    get_simmat_thetas(from_file, n_docs, thetas=None,
                                      path_to_model=path_model)
            else:
                print("No base path for the nruns models was provided.")
                return None
        else:
            if thetas is not None:
                sim_mat = get_simmat_thetas(from_file, n_docs, thetas)
            else:
                print("No thetas matrix was provided.")
                return None
        mean_sim += sim_mat
        quad_sim += np.square(sim_mat)
    if method == "normal":
        result = \
            np.mean(
                np.sqrt(np.maximum(0, quad_sim/n_runs - np.square(mean_sim/n_runs))))
    elif method == "percentage":
        mean_sim = mean_sim.flatten()
        quad_sim = quad_sim.flatten()
        std_devs = np.sqrt(np.maximum(
            0, quad_sim/n_runs - np.square(mean_sim/n_runs)))
        grt_means = np.argsort(mean_sim)[::-1]
        result = np.mean(std_devs[grt_means[:int(n_docs**2/10)]])
    else:
        print("Requested method not implemented")
        result = None
    return result


def get_repeated_elements_simmat_nruns(from_file, n_docs, n_runs, base_path=None, thetas=None):
    """Gets the repeated elements in top 10% of similarity matrix elements of nruns.

    Args:
    -----
        * from_file (boolean):                True if the thetas matrix is going to be 
                                              provided through a file.
        * n_docs (int):                       Nr of documents of the training data with which 
                                              the model was trained.
        * n_runs (int):                       Nr of times the traininig of the model has been 
                                              carried out. 
        * base_path (pathlib.Path, optional): Path where the "model.npz" files are located. Defaults to None.
        * thetas (darray, optional):          Document-topic distribution of the model 
                                              whose performance is being evaluated. Defaults to None.

    Returns:
    --------
        int: Nr of repeated elements.
    """
    for run in np.arange(n_runs):
        if from_file:
            if base_path:
                model_name = 'model_run_' + str(run)
                path_model = \
                    base_path.joinpath(model_name).joinpath('modelo.npz')
                sim_mat = \
                    get_simmat_thetas(from_file, n_docs, thetas=None,
                                      path_to_model=path_model)
            else:
                print("No base path for the nruns models was provided.")
                return None
        else:
            if thetas is not None:
                sim_mat = get_simmat_thetas(from_file, n_docs, thetas)
            else:
                print("No thetas matrix was provided.")
                return None
        grt_means = set(np.argsort(sim_mat.flatten())[
                        ::-1][:int(n_docs**2/10)].tolist())
        persistent = persistent.intersection(grt_means)
    result = len(persistent)
    return result


def get_sim_docs_frobenius(sim_mat_thetas_gt, sim_mat_thetas_model):
    diff_sims = sim_mat_thetas_model - sim_mat_thetas_gt
    frobenius_diff_sims = np.linalg.norm(diff_sims, 'fro')
    return frobenius_diff_sims


def get_sim_tops_frobenius(sim_mat_betas):
    simmat_pd = pd.DataFrame(sim_mat_betas)
    maxValues_rows = simmat_pd.max(axis=1)
    max_values_rows_sum = maxValues_rows.sum()
    return max_values_rows_sum
