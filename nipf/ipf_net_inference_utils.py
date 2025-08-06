"""
Utility/helper functions from the ipf-nework-inference git repo.

"""

import numpy as np
import random
from scipy.sparse import csr_matrix
from scipy.linalg import eig, eigh
from scipy import optimize
from scipy.stats import pearsonr, spearmanr
from scipy.stats import entropy
import statsmodels.api as sm
import argparse
import datetime
import itertools
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import os
import pandas as pd
import pickle
import random
# from scipy.sparse import csr_matrix
from scipy.linalg import eig
# from scipy import optimize
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from sklearn.metrics import pairwise_distances
# from sklearn.metrics.pairwise import cosine_similarity
import statsmodels.api as sm
import time

import torch
from ipf_utils import ipfn_wrapper
from nipf_utils import nipf2 as nipf

def do_ipf(X, p, q, num_iter=1000, start_iter=0, row_factors=None, col_factors=None, 
           eps=1e-8, return_all_factors=False, verbose=True):
    """
    X: initial matrix
    p: target row marginals
    q: target col marginals
    num_iter: number of iterations to run
    start_iter: which iteration to start at
                If start_iter > 0, then initial row_factors and col_factors must be provided
                Otherwise, we initialize to all ones
    row_factors: initial row factors
    col_factors: initial col factors
    eps: epsilon to check for convergence
    return_all_factors: whether to return row and column factors over all iterations or just 
                the final factors
    """
    assert X.shape == (len(p), len(q))
    if not np.isclose(np.sum(p), np.sum(q)):
        print('Warning: total row marginals do not equal total col marginals')
    # this allows us to continue from an earlier stopped iteration
    if start_iter > 0:
        assert row_factors is not None and col_factors is not None
        assert len(row_factors) == X.shape[0]
        assert len(col_factors) == X.shape[1]
        print(f'Starting from iter {start_iter}, received row and col factors')
        row_factors = row_factors.copy()
        col_factors = col_factors.copy()
    else:
        assert row_factors is None and col_factors is None
        row_factors = np.ones(X.shape[0])
        col_factors = np.ones(X.shape[1])
    if verbose:
        print(f'Running IPF for max {num_iter} iterations')
    
    all_row_factors = []
    all_col_factors = []
    all_est_mat = []
    row_errs = []
    col_errs = []
    for i in range(start_iter, start_iter+num_iter):
        if (i%2) == 0:  # adjust row factors
            row_sums = np.sum(X @ np.diag(col_factors), axis=1)
            # prevent divide by 0
            row_factors = p / np.clip(row_sums, 1e-8, None)
            # if marginals are 0, row factor should be 0
            row_factors[np.isclose(p, 0)] = 0
        else:  # adjust col factors
            col_sums = np.sum(np.diag(row_factors) @ X, axis=0)
            # prevent divide by 0
            col_factors = q / np.clip(col_sums, 1e-8, None)
            # if marginals are 0, column factor should be 0
            col_factors[np.isclose(q, 0)] = 0
        all_row_factors.append(row_factors)
        all_col_factors.append(col_factors)
     
        # get error from marginals
        est_mat = np.diag(row_factors) @ X @ np.diag(col_factors)
        all_est_mat.append(est_mat)
        row_err = np.sum(np.abs(p - np.sum(est_mat, axis=1)))
        col_err = np.sum(np.abs(q - np.sum(est_mat, axis=0)))
        row_errs.append(row_err)
        col_errs.append(col_err)
        if verbose:
            print('Iter %d: row err = %.4f, col err = %.4f' % (i, row_err, col_err))
        
        # check if converged
        if len(all_est_mat) >= 2:
            delta = np.sum(np.abs(all_est_mat[-1] - all_est_mat[-2]))
            if delta < eps:  # converged
                if verbose:
                    print(f'Converged; stopping after {i} iterations')
                break
            
        # check if stuck in oscillation
        if len(all_est_mat) >= 4:
            same1 = np.isclose(all_est_mat[-1], all_est_mat[-3]).all()
            same2 = np.isclose(all_est_mat[-2], all_est_mat[-4]).all()
            diff_consecutive = ~(np.isclose(all_est_mat[-1], all_est_mat[-2]).all())
            if same1 and same2 and diff_consecutive:
                if verbose:
                    print(f'Stuck in oscillation; stopping after {i} iterations')
                break                                
        
    if return_all_factors:  # return factors per iteration
        return i, np.array(all_row_factors), np.array(all_col_factors), row_errs, col_errs
    return i, row_factors, col_factors, row_errs, col_errs


####################################################################
# Compare IPF to Poisson regression
####################################################################
def run_poisson_experiment(X, p, q, Y=None, F=None, method='IRLS'):
    """
    Do Poisson regression on IPF inputs.
    X, p, q: IPF inputs.
    Y: response variable. If Y is not provided, we use max-flow algorithm to find appropriate Y.
    F: optional additional interation feature of size m x n.
    method: method used for fitting model. Default is 'IRLS' (iteratively reweighted least squares), 
        which is default for statsmodels. Another option is 'lbfgs' (default for sklearn).
    """
    assert X.shape == (len(p), len(q))
    assert (p > 0).all(), 'Row marginals must be positive for Poisson regression to converge'
    assert (q > 0).all(), 'Col marginals must be positive for Poisson regression to converge'
    m, n = X.shape
    
    # construct one-hot matrix representing row and col indices
    row_nnz, col_nnz = X.nonzero()
    nnz = len(row_nnz)
    csr_rows = np.concatenate([np.arange(nnz, dtype=int), np.arange(nnz, dtype=int)])
    csr_cols = np.concatenate([row_nnz, col_nnz+m]).astype(int)
    csr_data = np.concatenate([np.ones(nnz), np.ones(nnz)])
    onehots = csr_matrix((csr_data, (csr_rows, csr_cols)), shape=(nnz, m+n)).toarray()
    assert (np.sum(onehots, axis=1) == 2).all()  # each row should have two nonzero entries
    print('Constructed one-hot mat:', onehots.shape)
    
    # construct explanatory variables
    if F is not None:  # add interaction feature as f^alpha exp(-f * beta)
        feat = F[row_nnz, col_nnz].reshape(len(onehots), 1)
        log_feat = np.log(feat)
        explain_vars = np.concatenate([onehots, feat, log_feat], axis=1)
    else:
        explain_vars = onehots
    print('Constructed explanatory variables:', explain_vars.shape)

    # construct response variable
    if Y is None:  # get Y by running max-flow algorithm
        G, f_val, Y = test_ipf_convergence_from_max_flow(X, p, q, return_flow_mat=True)
        assert np.isclose(f_val, np.sum(p))  # IPF should converge
    assert Y.shape == X.shape
    assert (Y[X == 0] == 0).all()  # Y should inherit all zeros of X
    assert np.isclose(np.sum(Y, axis=1), p).all()  # Y should have target row marginals
    assert np.isclose(np.sum(Y, axis=0), q).all()  # Y should have target col marginals
    resp = Y[row_nnz, col_nnz]
    print('Constructed response variable:', resp.shape)
    
    ts = time.time()
    offset = np.log(X[row_nnz, col_nnz])  # include in linear model with coefficient of 1
    mdl = sm.GLM(resp, explain_vars, offset=offset, family=sm.families.Poisson())
    print('Initialized Poisson model [time=%.3fs]' % (time.time()-ts))
    ts = time.time()
    result = mdl.fit(method=method)
    print('Finished fitting model with statsmodels, method %s [time=%.3fs]' % (method, time.time()-ts))
    return mdl, result
    
# def visualize_ipf_vs_poisson_params(ipf_row_factors, ipf_col_factors, reg_coefs, reg_cis=None,
#                                     true_row_factors=None, true_col_factors=None, normalize=True,
#                                     log_ipf=False, xlim=None, ylim=None):
def visualize_ipf_vs_poisson_params(ipf_row_factors, ipf_col_factors,
                                    nipf_row_factors=None, nipf_col_factors=None,
                                    true_row_factors=None, true_col_factors=None, normalize=True,
                                    xaxis_label=None,
                                    log_ipf=False, xlim=None, ylim=None):
    """
    Plot IPF parameters vs truePoisson parameters. If log_ipf is True, log transform IPF parameters.
    If log_ipf is False, exponentiate the Poisson regression parameters.
    """
    m, n = len(ipf_row_factors), len(ipf_col_factors)
    # reg_row_coefs = np.exp(reg_coefs[:m])
    # reg_row_cis = np.exp(reg_cis[:m]) if reg_cis is not None else None
    # reg_col_coefs = np.exp(reg_coefs[m:])
    # reg_col_cis = np.exp(reg_cis[m:]) if reg_cis is not None else None
    if normalize:
        # normalize all factors by their mean, so that we can compare at y=x
        ipf_row_factors = ipf_row_factors / np.mean(ipf_row_factors)
        ipf_col_factors = ipf_col_factors / np.mean(ipf_col_factors)
        nipf_row_factors = nipf_row_factors / np.mean(nipf_row_factors) if nipf_row_factors is not None else None
        nipf_col_factors = nipf_col_factors / np.mean(nipf_col_factors) if nipf_col_factors is not None else None
        # reg_row_cis = reg_row_cis / np.mean(reg_row_coefs) if reg_row_cis is not None else None
        # reg_row_coefs = reg_row_coefs / np.mean(reg_row_coefs)
        # reg_col_cis = reg_col_cis / np.mean(reg_col_coefs) if reg_col_cis is not None else None
        # reg_col_coefs = reg_col_coefs / np.mean(reg_col_coefs)
        true_row_factors = true_row_factors / np.mean(true_row_factors) if true_row_factors is not None else None
        true_col_factors = true_col_factors / np.mean(true_col_factors) if true_col_factors is not None else None
    
    if log_ipf:
        ipf_row_factors = np.log(ipf_row_factors)
        ipf_col_factors = np.log(ipf_col_factors)
        ipf_row_label = '$\log(s^1_i)$'
        ipf_col_label = '$\log(s^2_j)$'
        # reg_row_coefs = np.log(reg_row_coefs)
        # reg_row_cis = np.log(reg_row_cis) if reg_row_cis is not None else None
        # reg_col_coefs = np.log(reg_col_coefs)
        # reg_col_cis = np.log(reg_col_cis) if reg_col_cis is not None else None
        # reg_row_label = '$\\theta_i$'
        # reg_col_label = '$\\theta_j$'
        true_row_factors = np.log(true_row_factors) if true_row_factors is not None else None
        true_col_factors = np.log(true_col_factors) if true_col_factors is not None else None    
        true_row_label = '$u_i$'
        true_col_label = '$-v_j$'
    else:
        ipf_row_label = '$s^1_i$'
        ipf_col_label = '$s^2_j$'  
        # reg_row_label = '$\exp(\\theta_i)$'
        # reg_col_label = '$\exp(\\theta_j)$'
        true_row_label = '$\exp(u_i)$'
        true_col_label = '$\exp(-v_j)$'
        
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.3)
    # plot row params 
    ax = axes[0]
    ax.scatter(true_row_factors, ipf_row_factors, color='tab:blue')
    # if reg_row_cis is not None:
    #     for i in range(m):
    #         ax.plot([ipf_row_factors[i], ipf_row_factors[i]], [reg_row_cis[i, 0], reg_row_cis[i, 1]], 
    #                 color='grey', alpha=0.5)
    ax.set_ylabel(f'{ipf_row_label} from IPF', color='tab:blue', fontsize=14)
    # color = 'black' if true_row_factors is None else 'tab:blue'
    if xaxis_label is not None:
        ax.set_xlabel(f'{true_row_label} from Poisson model', color='black', fontsize=14)
    ax.grid(alpha=0.2)
    if normalize:
        ax.plot(true_row_factors, true_row_factors, color='tab:red', label='y=x')
        ax.legend(loc='lower right', fontsize=14)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(labelsize=12)
    
    # plot column params
    ax = axes[1]
    ax.scatter(true_col_factors, ipf_col_factors, color='tab:blue', alpha=0.7)
    # if reg_col_cis is not None:
    #     for j in range(n):
    #         ax.plot([ipf_col_factors[j], ipf_col_factors[j]], [reg_col_cis[j, 0], reg_col_cis[j, 1]],
    #                 color='grey', alpha=0.5)
    ax.set_ylabel(f'{ipf_col_label} from IPF', color='tab:blue', fontsize=14)
    # color = 'black' if true_col_factors is None else 'tab:blue'
    if xaxis_label is not None:
        ax.set_xlabel(f'{true_row_label} from Poisson model', color='black', fontsize=14)
    # ax.set_xlabel(f'{true_col_label} from Poisson model', color='black', fontsize=14)
    ax.grid(alpha=0.2)
    if normalize:
        ax.plot(true_col_factors, true_col_factors, color='tab:red', label='y=x')
        ax.legend(loc='lower right', fontsize=14)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(labelsize=12)
    
    # add NIPF scaling factors in orange
    if nipf_row_factors is not None:
        ax_twin = axes[0].twinx()
        ax_twin.scatter(true_row_factors, nipf_row_factors, color='tab:orange', alpha=0.4)
        ax_twin.set_ylabel(f'{ipf_row_label} from NIPF', color='tab:orange', fontsize=14)
        ax_twin.set_xlim(axes[0].get_xlim())
        ax_twin.set_ylim(axes[0].get_ylim())
        ax_twin.tick_params(labelsize=12)
    if nipf_col_factors is not None:
        ax_twin = axes[1].twinx()
        ax_twin.scatter(true_col_factors, nipf_col_factors, color='tab:orange', alpha=0.4)
        ax_twin.set_ylabel(f'{ipf_col_label} from NIPF', color='tab:orange', fontsize=14)
        ax_twin.set_xlim(axes[1].get_xlim())
        ax_twin.set_ylim(axes[1].get_ylim())
        ax_twin.tick_params(labelsize=12)
    return fig, axes


def convert_bipartite_matrix_to_square_matrix(B):
    """
    Convert a bipartite matrix that is m x n into a square matrix 
    that is (m+n) x (m+n).
    """
    m, n = B.shape
    square = np.zeros((m+n, m+n))
    square[:m][:, m:] = B
    square[m:][:, :m] = B.T
    return square

def get_largest_eigenvalue_and_eigenvectors(square):
    """
    Return a square matrix's largest eigenvalue and corresponding left/right eigenvector.
    """
    is_symmetric = (np.isclose(square, square.T)).all()
    # get first eigenvalue and first eigenvectors
    ts = time.time()
    if is_symmetric:
        index = square.shape[0]-1  # ordered from smallest to largest, want last one
        w, u = eigh(square, subset_by_index=[index,index])  # eigh is much faster
        w = w[0]
        u = u.reshape(-1)
        v = u  # left and right eigenvectors are the same for symmetric matrix
    else:
        w, u, v = eig(square, left=True, right=True)
        largest = np.argmax(w)
        w = w[largest]
        u = u[:, largest]
        v = v[:, largest]
    print('Found eigenvalues and eigenvectors [time=%.3f]' % (time.time()-ts))
    return w, u, v


def compute_error_bound(X, row_factors, col_factors):
    """
    Compute relative bound (without constant) from Theorem 4.2 given X and network parameters.
    """
    m, n = X.shape
    means = np.diag(row_factors) @ X @ np.diag(col_factors)
    bound = np.sum(means)
    square = convert_bipartite_matrix_to_square_matrix(X)
    laplacian = np.diag(np.sum(square, axis=1)) - square
    # we can use eigh since laplacian is symmetric
    w = eigh(laplacian, subset_by_index=[1,1], eigvals_only=True)  # ordered from smallest to largest, want second-smallest
    w = w[0]
    bound = bound/(w**2)
    return bound

###################################################################
# Experiments with synthetic data
####################################################################
def generate_X(m, n, dist='uniform', seed=0, sparsity_rate=0, exact_rate=False, verbose=True):
    """
    Generate X based on kwargs.
    sparsity_rate: each entry is set to 0 with probability sparsity_rate.
    """
    np.random.seed(seed)
    assert dist in {'uniform', 'poisson'}
    if verbose:
        print(f'Sampling X from {dist} distribution')
    if dist == 'uniform':
        X = np.random.rand(m, n)
    elif dist == 'poisson':
        X = np.random.poisson(lam=10, size=(m,n))
    if sparsity_rate > 0:
        assert sparsity_rate < 1
        if exact_rate:
            random.seed(seed)
            num_zeros = int(sparsity_rate * (m*n))  # sample exactly this number of entries to set to 0
            pairs = list(itertools.product(range(m), range(n)))  # all possible pairs
            set_to_0 = random.sample(pairs, num_zeros)  # sample without replacement
            set_to_0 = ([t[0] for t in set_to_0], [t[1] for t in set_to_0])
        else:
            # set each entry to 0 with independent probability sparsity_rate
            set_to_0 = np.random.rand(m, n) < sparsity_rate
        X[set_to_0] = 0
        if verbose:
            print('Num nonzero entries in X: %d out of %d' % (np.sum(X > 0), m*n))
    return X

def generate_row_and_col_factors(m, n, seed=0, scalar=4):
    """
    Generate ground-truth row factors and column factors.
    """
    np.random.seed(seed)
    row_factors = np.random.rand(m) * scalar
    col_factors = np.random.rand(n) * scalar
    return row_factors, col_factors
    
def generate_hourly_network(X, row_factors, col_factors, model='basic', seed=0,
                            gamma=None, D=None, alpha=None, beta=None):
    """
    Generate hourly network based on time-aggregated network X and hourly row/column factors,
    and potentially other information. model defines which model is being used.
    """
    assert model in ['basic', 'exp', 'nb', 'mult', 'interaction']
    np.random.seed(seed)
    means = np.diag(row_factors) @ X @ np.diag(col_factors)  # original expected values
    if model == 'basic':  # biproportional Poisson
        Y = np.random.poisson(means)
    elif model == 'exp':  # exponential
        Y = np.random.exponential(means)
    elif model == 'nb':  # negative binomial
        assert gamma is not None
        n_successes = (gamma * means) / (1-gamma)
        Y = np.random.negative_binomial(n_successes, gamma)
    elif model == 'mult':  # multinomial
        N = np.random.poisson(np.sum(means))
        probs = means / np.sum(means)
        Y = np.random.multinomial(N, probs.flatten()).reshape(X.shape)
    else:
        assert model == 'interaction'
        assert D is not None and alpha is not None and beta is not None
        new_means = means * (D ** alpha) * np.exp(-D * beta)
        old_total = np.sum(means)
        new_total = np.sum(new_means)
        Y = np.random.poisson(new_means * (old_total / new_total))
    return Y

def generate_distance_mat(m, n, seed=0):
    """
    Generate positions and distance matrix.
    """
    np.random.seed(seed)
    row_pos = np.random.rand(m, 2)
    col_pos = np.random.rand(n, 2)
    dist = pairwise_distances(row_pos, col_pos)
    return dist
    

def do_ipf_and_eval(X, Y, normalized_expu, normalized_expv):
    """
    Run IPF and return 1) num iterations, 2) l2 distance to true network parameters, and 3) cosine similarity
    to true network.
    """
    i, row_factors, col_factors, row_errs, col_errs = do_ipf(X, Y.sum(axis=1), Y.sum(axis=0), verbose=False)
    if (row_errs[-1] + col_errs[-1]) > 1e-6:
        print('Warning: did not converge')
    row_factors = row_factors / np.mean(row_factors)
    col_factors = col_factors / np.mean(col_factors)
    row_diffs = row_factors - normalized_expu  # row-wise subtraction
    col_diffs = col_factors - normalized_expv
    l2 = np.sqrt(np.sum(row_diffs ** 2) + np.sum(col_diffs ** 2))
    est_mat = np.diag(row_factors) @ X @ np.diag(col_factors)
    cossim = np.dot(Y.flatten(), est_mat.flatten()) / (l2_norm(Y) * l2_norm(est_mat))
    return i, l2, cossim


####################################################################
# Experiments with bikeshare data from CitiBike
####################################################################
def prep_bikeshare_data_for_ipf(dt, timeagg='month', hours=None, networks=None,
                                 data_path='experiments/nnipf/out/bikeshare-202309.pkl',):
    """
    Prep bikeshare data for IPF.
    dt: datetime object, with year, month, day, and hour
    timeagg: how much time to aggregate over
    """
    assert (dt >= datetime.datetime(2023, 9, 1)) and (dt < datetime.datetime(2023, 10, 1))
    assert timeagg in ['month', 'week', 'day']
    print('Prepping bikeshare data for %s...' % datetime.datetime.strftime(dt, '%Y-%m-%d %H'))
    if hours is None or networks is None:
        with open(data_path, 'rb') as f:
            hours, networks = pickle.load(f)
    hour_idx = hours.index(dt)
    true_mat = networks[hour_idx].toarray()
    p = true_mat.sum(axis=1)  # row marginals
    q = true_mat.sum(axis=0)  # column marginals
    N = len(p)
    
    # get time-aggregated matrix
    if timeagg == 'month':
        start_idx = 0
        end_idx = len(networks)
    elif timeagg == 'week':
        week_idx = hour_idx // 168
        start_idx = 168 * week_idx
        end_idx = 168 * (week_idx + 1)
    else:
        day_idx = hour_idx // 24
        start_idx = 24 * day_idx
        end_idx = 24 * (day_idx + 1)
    X = 0
    for mat in networks[start_idx:end_idx]:
        X += mat
    nnz = X.count_nonzero()
    print('Aggregated to %s-level -> %d pairs (%.2f%%)' % (timeagg, nnz, 100 * nnz / (N*N)))
    X = X.toarray()
    return X, p, q, true_mat

def get_distances_between_stations():
    """
    Get pairwise distances between bike stations.
    """
    stations = pd.read_csv('experiments/nnipf/out/202309-bike-stations.csv').sort_values('station_num')
    locations = stations[['lat_mean', 'lng_mean']].values
    pairwise_dist = pairwise_distances(locations)
    return pairwise_dist

def dist_func(d, a, b):
    """
    Number of bike trips as a function of distance. Functional form from Navick and Furth (1994).
    """
    return d**a * np.exp(-d * b)

def fit_distance_function(X, max_dist=0.25):
    """
    Fit distance function on observed distances and number of trips between station pairs.
    """
    distances = get_distances_between_stations()
    # print(X.shape, distances.shape)
    assert distances.shape == X.shape
    # assert np.sum(np.isclose(distances, 0)) == distances.shape[0]
    min_nonzero = np.min(distances[distances > 0])
    distances = np.clip(distances, min_nonzero/2, None)  # fill zeros with epsilon
    
    mids = []
    trips = []
    interval = 0.001
    for start in np.arange(0, max_dist+interval, interval):
        end = start+interval
        in_range = (distances >= start) & (distances < end)
        mids.append(np.mean([start, end]))  # midpoint of interval
        trips.append(np.mean(X[in_range]))
    params, _ = curve_fit(dist_func, mids, trips)
    print('Estimated distance parameters: alpha=%.4f, beta=%.4f' % (params[0], params[1]))
    return distances, params
    
def l2_norm(mat):
    """
    Return L2 norm of a matrix.
    """
    return np.sqrt(np.sum(mat ** 2))

def eval_est_params(est_row_factors, est_col_factors, true_row_factors, true_col_factors):
    """
    Evaluate estimated row and column factors against true row and column factors.
    Returns normalized L2 distance, Pearson correlation, and cosine similarity.
    """
    est_row_factors = est_row_factors / np.mean(est_row_factors)
    est_col_factors = est_col_factors / np.mean(est_col_factors)
    true_row_factors = true_row_factors / np.mean(true_row_factors)
    true_col_factors = true_col_factors / np.mean(true_col_factors)
    
    row_diffs = est_row_factors - true_row_factors
    col_diffs = est_col_factors - true_col_factors
    norm_l2 = np.sqrt(np.sum(row_diffs ** 2) + np.sum(col_diffs ** 2))
    
    corr_row = pearsonr(true_row_factors, est_row_factors)[0]
    corr_col = pearsonr(true_col_factors, est_col_factors)[0]
    
    cossim_row = np.dot(true_row_factors, est_row_factors) / (l2_norm(true_row_factors) * l2_norm(est_row_factors))
    cossim_col = np.dot(true_col_factors, est_col_factors) / (l2_norm(true_col_factors) * l2_norm(est_col_factors))
    
    return norm_l2, corr_row, corr_col, cossim_row, cossim_col
    
def eval_est_mat(est_mat, real_mat, verbose=True):
    """
    Evaluate distance between real matrix and estimated matrix.
    """
    if not np.isclose(est_mat.sum(), real_mat.sum()):
        print('Warning: matrices do not have the same total, off by %.3f' % np.abs(est_mat.sum()-real_mat.sum()))
    if not np.isclose(real_mat.sum(axis=1), est_mat.sum(axis=1)).all():
        print('Warning: row marginals don\'t match')
    if not np.isclose(real_mat.sum(axis=0), est_mat.sum(axis=0)).all():
        print('Warning: col marginals don\'t match')
    norm_l2 = l2_norm(est_mat - real_mat) / l2_norm(real_mat)
    if verbose:
        print('Normalized L2 distance', norm_l2)
    corr = pearsonr(real_mat.flatten(), est_mat.flatten())
    if verbose:
        print('Pearson corr', corr)
    cossim = np.dot(real_mat.flatten(), est_mat.flatten()) / (l2_norm(real_mat) * l2_norm(est_mat))
    if verbose:
        print('Cosine sim', cossim)
    # include the KL divergence
    # normalize matrices to sum to 1
    real_mat_norm = real_mat / np.sum(real_mat)
    est_mat_norm = est_mat / np.sum(est_mat)
    kl_div = entropy(real_mat_norm.flatten(), est_mat_norm.flatten())
    if verbose:
        print('KL divergence', kl_div)
    return norm_l2, corr, cossim, kl_div

def run_bikeshare_ipf_experiment(dt, timeagg='month', use_gravity=False, max_iter=1000):
    """
    Run IPF experiment on bikeshare data.
    """
    X, p, q, true_mat = prep_bikeshare_data_for_ipf(dt, timeagg)
    if use_gravity:
        print('Fitting IPF gravity model: replacing X with distance mat')
        distances, params = fit_distance_function(X)
        X = dist_func(distances, params[0], params[1])
        
    ts = time.time()
    # ipf_out = do_ipf(X, p, q, num_iter=max_iter)
    # ipf_out = ipfn_wrapper(X, [p, q])
    _, scale_factors, _ = nipf(torch.tensor(X), [torch.tensor(p), torch.tensor(q)], epochs=20000, lr=0.1)
    print('Finished IPF: time=%.2fs' % (time.time()-ts))    

    # row_factors, col_factors = ipf_out[1], ipf_out[2]
    ipf_out = [
        scale.cpu().numpy() for scale in scale_factors
    ]
    row_factors, col_factors = ipf_out[0], ipf_out[1]
    est_mat = np.diag(row_factors) @ X @ np.diag(col_factors)
    print('Comparing real matrix and estimated matrix')
    eval_est_mat(est_mat, true_mat)
    
    if use_gravity:
        fn = 'experiments/nnipf/out/ipf-output/bikeshare_%s_gravity_%s.pkl' % (timeagg, dt.strftime('%Y-%m-%d-%H'))
    else:
        fn = 'experiments/nnipf/out/ipf-output/bikeshare_%s_%s.pkl' % (timeagg, dt.strftime('%Y-%m-%d-%H'))
    print('Saving results in', fn)
    with open(fn, 'wb') as f:
        pickle.dump(ipf_out, f)
        
        
def run_bikeshare_all_hours_in_day(dt, timeagg='month', use_gravity=False, max_iter=1000):
    """
    Outer function to run IPF for all hours in a given day.
    """
    print('Running IPF on bikeshare data, all hours on %s...' % dt.strftime('%Y-%m-%d'))
    for hr in range(24):
        curr_dt = datetime.datetime(year=dt.year, month=dt.month, day=dt.day, hour=hr)
        if use_gravity:
            out_file = 'experiments/nnipf/out/ipf-output/bikeshare_%s_gravity_%s.out' % (timeagg, curr_dt.strftime('%Y-%m-%d-%H'))
        else:
            out_file = 'experiments/nnipf/out/ipf-output/bikeshare_%s_%s.out' % (timeagg, curr_dt.strftime('%Y-%m-%d-%H'))
        cmd = f'nohup python -u experiments_with_data.py ipf_single_hour bikeshare {dt.year} {dt.month} {dt.day} --hour {hr} --timeagg {timeagg} --ipf_gravity {int(use_gravity)} --max_iter {max_iter} > {out_file} 2>&1 &'
        print(cmd)
        os.system(cmd)
        time.sleep(1)
        

def baseline_no_mat(p, q):
    """
    Baseline where we ignore time-aggregated matrix and only use marginals.
    """
    outer_prod = np.outer(p, q) * 1.0
    outer_prod /= np.sum(outer_prod)
    outer_prod *= np.sum(p)  # scale to sum to marginal total
    assert np.isclose(np.sum(outer_prod, axis=1), p).all()
    assert np.isclose(np.sum(outer_prod, axis=0), q).all()
    return outer_prod

def baseline_no_col(X, p):
    """
    Baseline where we ignore column marginals and only use X and p.
    """
    row_sums = X.sum(axis=1)
    row_factors = p / row_sums
    row_factors[row_sums == 0] = 0
    est_mat = np.diag(row_factors) @ X
    assert np.isclose(np.sum(est_mat, axis=1), p).all()
    return est_mat

def baseline_no_row(X, q):
    """
    Baseline where we ignore row marginals and only use X and q.
    """
    col_sums = X.sum(axis=0)
    col_factors = q / col_sums
    col_factors[col_sums == 0] = 0
    est_mat = X @ np.diag(col_factors)
    assert np.isclose(np.sum(est_mat, axis=0), q).all()
    return est_mat

def baseline_scale_mat(X, total):
    """
    Baseline where we rescale X so that its total is equal to the hourly total.
    """
    curr_total = X.sum()
    est_mat = X * total / curr_total
    assert np.isclose(est_mat.sum(), total)
    return est_mat

def evaluate_results_on_bikeshare(dt, methods=None):
    """
    Evaluate results from different methods over 24 hours of bikeshare data for a given day.
    """
    assert (dt >= datetime.datetime(2023, 9, 1)) and (dt < datetime.datetime(2023, 10, 1))
    if methods is None:
        methods = ['ipf_month', 'ipf_week', 'ipf_day', 'gravity',
                   'baseline_no_mat', 'baseline_no_col', 'baseline_no_row',
                   'baseline_scale_month', 'baseline_scale_week', 'baseline_scale_day']
    with open('experiments/nnipf/out/bikeshare-202309.pkl', 'rb') as f:
        hours, networks = pickle.load(f)
    
    Xs = {}
    Xs['month'] = prep_bikeshare_data_for_ipf(dt, timeagg='month', hours=hours, networks=networks)[0]
    Xs['week'] = prep_bikeshare_data_for_ipf(dt, timeagg='week', hours=hours, networks=networks)[0]
    Xs['day'] = prep_bikeshare_data_for_ipf(dt, timeagg='day', hours=hours, networks=networks)[0]
    if 'gravity' in methods:
        distances, params = fit_distance_function(Xs['month'])
        Xs['distance'] = dist_func(distances, params[0], params[1])
    l2_dict = {m:[] for m in methods}
    pearson_dict = {m:[] for m in methods}
    cosine_dict = {m:[] for m in methods}

    for hr in range(24):
        curr_dt = datetime.datetime(year=dt.year, month=dt.month, day=dt.day, hour=hr)
        print('\n', curr_dt.strftime('%Y-%m-%d-%H'))
        hour_idx = hours.index(curr_dt)
        true_mat = networks[hour_idx].toarray()
        p = true_mat.sum(axis=1)  # row marginals
        q = true_mat.sum(axis=0)  # column marginals
        for m in methods:
            est_mat = _get_estimated_matrix(Xs, p, q, curr_dt, m)
            if est_mat is not None:
                l2, (r,_), cossim = eval_est_mat(est_mat, true_mat, verbose=False)
                print(m, 'L2=%.3f, Pearson r=%.3f, cosine sim=%.3f' % (l2, r, cossim))
            else:
                l2, r = np.nan, np.nan
            l2_dict[m].append(l2)
            pearson_dict[m].append(r)
            cosine_dict[m].append(cossim)
    return l2_dict, pearson_dict, cosine_dict
                
                
def _get_estimated_matrix(Xs, p, q, dt, method):
    """
    Helper method to get the estimated matrix for a given hour.
    """
    if method.startswith('ipf_') or method == 'gravity':
        if method.startswith('ipf_'):
            timeagg = method.split('_', 1)[1]
            fn = 'experiments/nnipf/out/ipf-output/bikeshare_%s_%s.pkl' % (timeagg, dt.strftime('%Y-%m-%d-%H'))
        else:
            timeagg = 'distance'
            fn = 'experiments/nnipf/out/ipf-output/bikeshare_month_gravity_%s.pkl' % dt.strftime('%Y-%m-%d-%H')
        if os.path.isfile(fn):
            with open(fn, 'rb') as f:
                ipf_out = pickle.load(f)
            # row_factors, col_factors = ipf_out[1], ipf_out[2]
            row_factors, col_factors = ipf_out[0], ipf_out[1]
            X = Xs[timeagg]
            est_mat = np.diag(row_factors) @ X @ np.diag(col_factors)
        else:
            print('File is missing:', fn)
            est_mat = None
    elif method.startswith('baseline_scale_'):
        timeagg = method.rsplit('_', 1)[1]
        X = Xs[timeagg]
        est_mat = baseline_scale_mat(X, np.sum(p))
    else:
        assert method.startswith('baseline_no_')
        if method == 'baseline_no_mat':
            est_mat = baseline_no_mat(p, q)
        elif method == 'baseline_no_col':
            est_mat = baseline_no_col(Xs['month'], p)
        else:
            assert method == 'baseline_no_row'
            est_mat = baseline_no_row(Xs['month'], q)
    return est_mat
