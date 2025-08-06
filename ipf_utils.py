#==========================================================
# Author: William O. Agyapong
# Purpose: core implementation codes for PhD Dissertation
# Created on: 04-20-2024
# Modifed on: 05-02-2024
#==========================================================

import numpy as np
import pandas as pd
import itertools
import functools
import pyipu


#---------- Classical two-way IPF
def two_way_ipf(X, u, v):
    M = X.copy()
    M = M.astype(float)
    pass
    # rowwise adjustment

# Randomly split the original data into reference and target data
def split_data(df, ref_size=0.1, random_state=123):
    # Shuffle the DataFrame and split it
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_index = int(len(df) * ref_size)
    reference_data = df_shuffled[:split_index]
    target_data = df_shuffled[split_index:]
    return reference_data, target_data

from scipy.stats import chi2_contingency, entropy
# from sklearn.metrics import mutual_info_score

def generate_contingency_table_2d(m, n, N, alpha=None, beta=None):
    if alpha is not None:
        # Strong dependence
        p = np.full((m, n), (1 - alpha) / (m * n))
        for i in range(min(m, n)):
            p[i, i] += alpha
    elif beta is not None:
        # Weak dependence
        p = np.full((m, n), (1 - beta) / (m * n))
        for i in range(min(m, n)):
            p[i, i] += beta
    else:
        # No dependence
        p = np.full((m, n), 1 / (m * n))
    
    p = p / p.sum()  # Ensure the probabilities sum to 1 (normalization)
    counts = np.random.multinomial(N, p.flatten()).reshape((m, n))
    return counts



def generate_contingency_table(dimensions, N, alpha=None, beta=None):
    """
    Generates a contingency table of arbitrary dimensions with specified dependence.

    Parameters:
    - dimensions (tuple): Dimensions of the contingency table (e.g., (4, 4, 4) for a 4x4x4 table).
    - N (int): Total count to be distributed in the table.
    - alpha (float, optional): Parameter controlling the strength of strong dependence.
    - beta (float, optional): Parameter controlling the strength of weak dependence.

    Returns:
    - table (ndarray): Generated contingency table.
    """
    # Initialize probabilities with a uniform distribution
    p = np.full(dimensions, 1 / np.prod(dimensions))
    
    if alpha is not None or beta is not None:
        # Apply dependence enhancement to slices
        dependence_strength = alpha if alpha is not None else beta
        boost_indices = [] 
        for i in range(len(dimensions)): 
            idx = np.arange(dimensions[i]) 
            boost_indices.append(np.meshgrid(*[idx]*len(dimensions), indexing='ij'))
        
        boost_indices = np.array(boost_indices)
        diagonal_boost_indices = np.diagonal(boost_indices, axis1=0, axis2=1)
        p[tuple(diagonal_boost_indices)] += dependence_strength
        p = p / p.sum()  # Normalize to ensure probabilities sum to 1
    
    # Generate the contingency table using the multinomial distribution
    counts = np.random.multinomial(N, p.flatten()).reshape(dimensions)
    
    return counts


def compute_dependence_measures(table):
    chi2, p, dof, expected = chi2_contingency(table)
    cramers_v = np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1)))
    # mutual_info = mutual_info_score(None, None, contingency=table)

    # consider using these measures: from info_measures import total_correlation
    # import dit
    # from dit.multivariate import total_correlation
    
    return chi2, cramers_v

def summarize_sim_results(results):
    summary = {}
    for key in results:
        measures = np.array(results[key])
        mean_measures = np.mean(measures, axis=0)
        std_measures = np.std(measures, axis=0)
        summary[key] = {
            "mean_chi2": mean_measures[0],
            "std_chi2": std_measures[0],
            "mean_cramers_v": mean_measures[1],
            "std_cramers_v": std_measures[1]
            # "mean_mutual_info": mean_measures[2],
            # "std_mutual_info": std_measures[2],
        }
    return summary

def ipf_2d(initial_table, row_margins, col_margins, max_iter=1000, tol=1e-6):
    table = initial_table.copy().astype(float)
    for _ in range(max_iter):
        # Adjust rows
        row_factors = row_margins / table.sum(axis=1)
        table = (table.T * row_factors).T
        
        # Adjust columns
        col_factors = col_margins / table.sum(axis=0)
        table *= col_factors
        
        # Check for convergence
        if np.allclose(row_margins, table.sum(axis=1), atol=tol) and np.allclose(col_margins, table.sum(axis=0), atol=tol):
            break
    return table 
    
    

#----- Higher-order IPF: 
# Credit to the Data Science Topics website for providing the Python
# implementation codes: https://datascience.oneoffcoder.com/ipf-ii.html

def get_target_marginals(d):
    factors = list(d.keys()) # get the control variables
    targets = [sorted([(k2, v2) for k2, v2 in v.items()]) for k, v in d.items()]
    # error here because Numpy array doesn't accept lists of different sizes
    # targets = np.array([[v for _, v in item] for item in targets])
    targets = [[v for _, v in item] for item in targets]
    return factors, targets

def get_table(df, targets):
    factors, target_marginals = get_target_marginals(targets)

    cross_tab = pd.crosstab(df[factors[0]], [df[c] for c in factors[1:]])
    shape = tuple([df[c].unique().shape[0] for c in factors])
    table = cross_tab.values.reshape(shape)

    return factors, target_marginals, table


# 
def c_table(df, target_vars, target_df=None, dropna=False):
    # the contingency table for the weighted data becomes sparse, with sparcity
    # increasing with increasing number of controls in the IPF
    # Hence dropna = False to preserve the dimension/categories
    # with no data
    cross_tab = pd.crosstab(df[target_vars[0]],
    [df[c] for c in target_vars[1:]], dropna=dropna)
    if target_df is not None:
        # Help preserve all the categories in the target on the reference
        shape = tuple([target_df[c].unique().shape[0] for c in target_vars])
    else:
        shape = tuple([df[c].unique().shape[0] for c in target_vars])
    table = cross_tab.values.reshape(shape)

    return table

# get the target margins
def ipf_target_in(target, reference, columns, path=None, save_to_df=True):
    target_ipf = target.copy()
    ref_ipf = reference.copy()
    margins = {}
    for col in columns:
        unique, counts = np.unique(target_ipf[col], return_counts=True)
        # counts = counts/counts.sum() # Doesn't impact results drammatically, seems to help rather
        # Remove values from target that are not in source
        unique_ref = np.unique(ref_ipf[col])
        unique_cleaned = list(unique)
        counts_cleaned = list(counts)
        for value in unique:
            if value not in unique_ref:
                idx = unique_cleaned.index(value)
                unique_cleaned.pop(idx)
                counts_cleaned.pop(idx)
        df = pd.DataFrame(dict(zip(unique_cleaned, counts_cleaned)), index=[0])
        # Put 0 where data in source not in target
        # print(df)
        # print(counts_cleaned)
        # margins.append(np.array(counts_cleaned))
        # print(counts_cleaned)
        margins[col] = counts_cleaned
        # for value in unique_ref:
        #     if value not in unique:
        #         # df[value] = 0
        #         # idx = unique_cleaned.index(value)
        #         pass
        # df.to_csv(f"{path}/{col}.csv", index=False)
    return margins
# usage
# if __name__ == "__main___":
#     f, u, X = get_table(df, {
#     'race': {'white': 5800, 'other': 4200},
#     'age': {'minor': 2800, 'adult': 7200},
#     'gender': {'male': 4900, 'female': 5100}
#     })


def get_coordinates(M):
    return list(itertools.product(*[list(range(n)) for n in M.shape]))

def get_marginals(M, i):
    """
    M: reference table
    i: control variable index from 0 to (total - 1)
    """
    coordinates = get_coordinates(M)

    key = lambda tup: tup[0]
    counts = [(c[i], M[c]) for c in coordinates]
    counts = sorted(counts, key=key)
    counts = itertools.groupby(counts, key=key)
    counts = {k: sum([v[1] for v in g]) for k, g in counts}

    return counts

def get_all_marginals(M):
    
    # return np.array([[v for _, v in get_marginals(M, i).items()]
    #                  for i in range(len(M.shape))
    #                  ]) # doesn't work if factor lengths differ across variables
    return [[v for _, v in get_marginals(M, i).items()]
                     for i in range(len(M.shape))
                     ]

def compute_marginals(table):
    """
    Compute the marginal totals along each dimension of an n-dimensional contingency table.
    
    Parameters:
    table (np.ndarray): n-dimensional contingency table.
    
    Returns:
    list of np.ndarray: Marginal totals along each dimension.
    """
    marginals = []
    for axis in range(table.ndim):
        marginals.append(np.sum(table, axis=tuple(i for i in range(table.ndim) if i != axis)))
    return marginals

def compute_margins_from_df(reference_df, target_df, normalize=False):
    """Compute marginal distributions from given reference and target data frames
    """
    ref_margins = {}
    target_margins = {}

    for col in target_df.columns:
        # Calculate marginal frequencies
        ref_freq = reference_df[col].value_counts(normalize=normalize).sort_index()
        target_freq = target_df[col].value_counts(normalize=normalize).sort_index()
        
        # Ensure all categories from the target marginal are in the reference frequencies
        ref_freq = ref_freq.reindex(target_freq.index, fill_value=0)
        
        # Convert reference and target frequencies to arrays
        ref = ref_freq.values
        target = target_freq.values
        
        # Store frequencies
        ref_margins[col] = ref
        target_margins[col] = target
    return ref_margins, target_margins


def modify_margins(margins, feature_names, modification_type='skew', skewness_factor=1.5, noise_level=0.1, fat_tail_factor=1.2):
    """
    Modify the margins of a contingency table according to the specified modification type.

    Parameters:
    - margins (list of ndarrays): Original margins of the contingency table.
    - modification_type (str): The type of modification to apply. Options are 'skew', 'uniform', 'fat_tail', 'thin_tail', 'perturb'.
    - skewness_factor (float): Factor to control the skewness.
    - noise_level (float): Level of noise to add or subtract (as a percentage).
    - fat_tail_factor (float): Factor to control the modification of tails (fat or thin).

    Returns:
    - modified_margins (list of ndarrays): The modified margins.
    """
    # modified_margins = []
    modified_margins = {}
    i = 0
    for margin in margins:
        if modification_type == 'skew':
            # Apply skewness by raising the margin to a power (skewness_factor)
            modified_margin = np.power(margin, skewness_factor)
            modified_margin = modified_margin / modified_margin.sum() * margin.sum()  # Normalize to the original sum
        
        elif modification_type == 'uniform':
            # Create uniform margins
            modified_margin = np.full_like(margin, margin.sum() / len(margin))
        
        elif modification_type == 'fat_tail':
            # Increase the tails of the margin distribution
            midpoint = len(margin) // 2
            lower_tail = margin[:midpoint] * fat_tail_factor
            upper_tail = margin[midpoint:] * fat_tail_factor
            modified_margin = np.concatenate((lower_tail, upper_tail))
            modified_margin = modified_margin / modified_margin.sum() * margin.sum()  # Normalize to the original sum
        
        elif modification_type == 'thin_tail':
            # Decrease the tails of the margin distribution
            midpoint = len(margin) // 2
            lower_tail = margin[:midpoint] / fat_tail_factor
            upper_tail = margin[midpoint:] / fat_tail_factor
            modified_margin = np.concatenate((lower_tail, upper_tail))
            modified_margin = modified_margin / modified_margin.sum() * margin.sum()  # Normalize to the original sum
        
        elif modification_type == 'perturb':
            # Add/subtract noise to each bin of the margin
            perturbation = np.random.uniform(-noise_level, noise_level, size=margin.shape)
            modified_margin = margin * (1 + perturbation)
            modified_margin = modified_margin / modified_margin.sum() * margin.sum()  # Normalize to the original sum
        elif modification_type =='none':
            modified_margin = margin
        else:
            raise ValueError("Invalid modification_type specified.")
        
        # modified_margins.append(modified_margin)
        modified_margins[feature_names[i]] = modified_margin
        i = i + 1
    
    return modified_margins


def get_counts(M, i):
    coordinates = get_coordinates(M)

    key = lambda tup: tup[0]
    counts = [(c[i], M[c], c) for c in coordinates]
    counts = sorted(counts, key=key)
    counts = itertools.groupby(counts, key=key)
    counts = {k: [(tup[1], tup[2]) for tup in g] for k, g in counts}

    return counts

def update_values(M, i, u):
    marg = get_marginals(M, i)
    vals = get_counts(M, i)

    d = [[(c, n*u[k] / marg[k]) for n, c in v] for k, v in vals.items()]
    d = itertools.chain(*d)
    d = list(d)
    # print(d)

    return d

def ipf_update(M, u, convergence_metric="l2"):
    for i in range(len(M.shape)):
        values = update_values(M, i, u[i])
        for idx, v in values:
            M[idx] = v
            # print(M[idx])

    o = get_all_marginals(M)
    d = get_deltas(o, u, convergence_metric)

    return M, d

def get_deltas(o, t, convergence_metric="l2"):
    # return np.array([np.linalg.norm(o[r] - t[r], 2) for r in range(o.shape[0])])
    if convergence_metric=="l2":
        return np.array([np.linalg.norm(np.array(o[r]) - np.array(t[r]), 2) for r in range(len(o))])
    elif convergence_metric=="l1":
        return np.array([np.linalg.norm(np.array(o[r]) - np.array(t[r]), 1) for r in range(len(o))])

def ipf(X, u, convergence_metric="l2", max_iters=1000, epsilon=0, zero_threshold=1e-10, convergence_threshold=3, debug=False):
    """
    ------------
    Parameters:
    epsilon: set to a non-zero small number to avoid division by zero when some reference margins are zero.
    """
    M = X.copy()
    
    # change data type to a float to help resolve the decimal truncation bug:
    # decimal values assigned into M in the update step become integers.
    M = M.astype(float)
    
    d_prev = np.zeros(len(M.shape))
    count_zero = 0
    num_iters = 0

    for _ in range(max_iters):
        # track # iterations
        num_iters += 1
        
        # update 
        M, d_next = ipf_update(M, u, convergence_metric)
        if convergence_metric == "l2":
            d = np.linalg.norm(d_prev - d_next, 2)
        elif convergence_metric == "l1":
            d = np.linalg.norm(d_prev - d_next, 1)
       
        if d < zero_threshold: # better results with is approach??
            count_zero += 1

        if debug:
            print(f'iter {num_iters}: ',','.join([f'{v:.5f}' for v in d_next]), d)
        d_prev = d_next
        

        if count_zero >= convergence_threshold:
            if debug:
                print('IPF converged after', num_iters, 'iterations.')
            break
        # if d < convergence_threshold: # doesn't work so well 
        #     print('IPF converged after', num_iters, 'iterations.')
        #     break
    
    # w = M/M.sum()
    return M

# new ipf with no max iterations
def ipf_new(X, u, convergence_metric = "l2",zero_threshold=0.0001, debug=True):
    M = X.copy()

    d_prev = np.zeros(len(M.shape))
    count_iters = 0
    d = 10
    while d > zero_threshold:
        count_iters += 1
        M, d_next = ipf_update(M, u,convergence_metric)
        if convergence_metric == 'l1':
            d = np.linalg.norm(d_prev - d_next, 1)
        elif convergence_metric == 'l2':
            d = np.linalg.norm(d_prev - d_next, 2)

        if debug:
            print(f'iter {count_iters}: ',','.join([f'{v:.5f}' for v in d_next]), d)
        d_prev = d_next

    
    # w = M/M.sum()
    return M


# A wrapper function for ipfn
from ipfn import ipfn
def ipfn_wrapper(X, target_margins, convergence_rate=1e-6, rate_tolerance = 1e-8, other_args={}):
    # other_args={}
    dimensions = [[i] for i in range(len(target_margins))]
    # IPF = ipfn.ipfn(X, target_margins , dimensions, **other_args) # , convergence_rate=1e-10
    IPF = ipfn.ipfn(X, target_margins , dimensions, convergence_rate=convergence_rate, rate_tolerance=rate_tolerance, **other_args)
    m = IPF.iteration()
    return m

# get the target margins for the IPU algorithm
# Each key in the dictionary maps to a data frame with values as columns and a single row
# of observation for the counts/frequencies
def ipu_target_in(target, source, columns, path=None, save_to_df=True):
    """Generate marginal target distribution input for IPU
    """
    target_ipf = target.copy()
    source_ipf = source.copy()
    margins = {}
    for col in columns:
        unique, counts = np.unique(target_ipf[col], return_counts=True)
        # Remove values from target that are not in source
        unique_source = np.unique(source_ipf[col])
        unique_cleaned = list(unique)
        counts_cleaned = list(counts)
        for value in unique:
            if value not in unique_source:
                idx = unique_cleaned.index(value)
                unique_cleaned.pop(idx)
                counts_cleaned.pop(idx)
        df = pd.DataFrame(dict(zip(unique_cleaned, counts_cleaned)), index=[0])
        # Put 0 where data in source not in target
        for value in unique_source:
            if value not in unique:
                df[value] = 0
       
        # create this margins dict for use with pyipu
        df.columns = df.columns.astype(str)
        margins[col] = df
    return margins

# A wrapper function for IPU population synthesizer
def ipu_syn(primary_seed, primary_targets, **kwargs):
    result = pyipu.ipu(primary_seed=primary_seed, primary_targets=primary_targets,  **kwargs)
    syn = pyipu.synthesize(result['weight_tbl']).drop(['id', 'new_id', 'geo_all'], axis=1)
    return syn, result['weight_tbl']

def std_ipf(ref_set, target_set, selected_features=None, conv_rate=1e-6, tol=1e-8, ipf_args={}):
    if selected_features is None:
        selected_features = list(target_set.columns)
    # Split data
    # Compute contingency tables for train and testing sets
    # X_ref = c_table(ref_set, selected_features, target_set) # preserves dimensions
    X_ref = c_table(ref_set, selected_features) 
    X_target = c_table(target_set, selected_features)
    # ref_margins = ipf_utils.compute_marginals(X_ref) # sample margins
    target_margins = compute_marginals(X_target) # True/Population margins

    # target_margins = ipf_target_in(target_set, ref_set, selected_features)
    
    # Run the standard IPF on the data
    X_ref_adjusted = ipfn_wrapper(X_ref, target_margins, convergence_rate=conv_rate, rate_tolerance=tol, other_args=ipf_args) # , other_args={"max_iteration":10000}
    # X_ref_adjusted = ipf(X_ref, target_margins) # convergence issues with zero margins

    # Resampling (synthesizing) data
    w = (X_ref_adjusted/X_ref_adjusted.sum())/(X_ref+1e-20) # 
    # w = (X_ref_adjusted)/(X_ref+1e-20) # Any effect in using the unnormalized population values? Doesn't make any obvious difference.
    # w = (X_ref_adjusted/X_ref_adjusted.sum())/((X_ref/X_ref.sum())+1e-20) # how about normalizing each table values to probs? Doesn't appear to cause any difference.
    syn_ref_set = get_samples(ref_set, selected_features, w, n=target_set.shape[0])
    weights = get_sampling_weights(ref_set, selected_features, w)
    return syn_ref_set, X_ref_adjusted, weights


## Generate sampling weights for each sample in the source data
# This function distributes the weights from the adjusted contingency table to the samples/respondents
# Use this when w is just weights = M/M.sum()
    # get_filters = lambda df, fields, values: [df[f] == v for f, v in zip(fields, values)]
    # get_totals = lambda df, fields, values: df[functools.reduce(lambda a, b: a & b, get_filters(df, fields, values))].shape[0]
    # return {k: v / get_totals(df, f, k) for k, v in zip(list(itertools.product(*[sorted(df[c].unique()) for c in f])), np.ravel(w))}

    # Use this when w is (M/M.sum())/X, the normalized adjusted table scaled by original table
    # This approach is even way faster
    # Note that (M)/X, can also be used, that is M need not be normalized
    # return {k: v for k, v in zip(list(itertools.product(*[sorted(df[c].unique()) for c in f])), np.ravel(w))}
def get_sampling_weights(df, f, w):
    weights =  {k: v for k, v in zip(list(itertools.product(*[sorted(df[c].unique()) for c in f])), np.ravel(w))}
    # print(f"Sampling weights: {weights}")
    w = df.apply(lambda r: weights[tuple([r[c] for c in f])], axis=1)
    return w
    
def get_samples(df, f, w_mat, n=10_000):
    """
    w_mat: is the weight matrix/table based on the adjusted contingency table. Note that this is not the raw ipf-adjusted table, but the normalized table (T_hat/T).
    """
    # weights = get_sampling_weights(df, f, w)
    # s = df.apply(lambda r: weights[tuple([r[c] for c in f])], axis=1)
    w = get_sampling_weights(df, f, w_mat)
    # print(s)
    #TODO Consider attaching sampling weights to the data
    return df.sample(n=n, replace=True, weights=w) 

def synthesize(df, weights, n=None):
    """Reweight input data to match the weights provided via sampling with replacement. 

    Parameters
    ---------
    df: sample data set as a pandas data frame
    weights: a vector of sampling weights of length equal to df.shape[0]. A pandas series. The sum of weights is assumed to be equal to the underlying population size.

    Return
    ------
    A pandas data frame
    """
    #TODO valid input, length of weights
    n = n if n is not None else int(np.round(weights.sum(),0))
    return df.sample(n=n, replace=True, weights=weights) 

def std_ipf_syn(df, X, X_adjusted, selected_features=None, epsilon=1e-10):
    # epsilon: a small number that aleviates division by zero problems
    if selected_features is None:
        selected_features = list(df.columns)

    # w = (X_adjusted/X_adjusted.sum())/(X+epsilon) # 
    w = (X_adjusted)/(X+epsilon) # 
    sampled_df = get_samples(df, selected_features, w, n=df.shape[0])
    #TODO Consider attaching sampling weights to the data
    return sampled_df 


#---------- Sequential IPF ----------
import random 
def group_features(selected_features, n_group=2, random_seed=None):
    # Shuffle feature order before splitting into groups
    if random_seed is not None:
        random.seed(random_seed)
        
    selected_features  = selected_features.copy()
    random.shuffle(selected_features)

    group_size = len(selected_features)//n_group # determine group size
    rem = len(selected_features)%group_size
    # itr = int(len(selected_features)/group_size)
    feature_seq = [] # sequence of 
    i = 0
    for k in range(n_group):
    # feature_seq.append(selected_features[i:(k+1)*group_size])
        if k == (n_group-1): # We've hit the end of the loop
            if rem == 1: # add to the last group
                feature_seq.append(selected_features[-(group_size+1):])
            elif rem !=0 and rem!=1: # put remaining into a new group
                feature_seq.append(selected_features[i:(k+1)*group_size])
                feature_seq.append(selected_features[-rem:])
            elif rem == 0:
                feature_seq.append(selected_features[i:(k+1)*group_size]) # extract the last group
        else:
            feature_seq.append(selected_features[i:(k+1)*group_size]) # extract and store all other previous groups

        i = (k+1)*group_size # update i to the previous left slicer
    return feature_seq

def group_features_dep():
    # ---- This grouping uses overlapping groups
    columns = list(df.columns)
    step_size = 2 # gives groups of 3s (0, 1, 2)
    print(columns, '\n')
    for i in range(len(columns) - step_size):
        selected_columns = columns[i:i+step_size+1]
        # print(f"Iteration {i + 1}:")
        if i < (len(columns)-(step_size+1)):
            print(selected_columns)
        else:
            print(f"Last iteration: {selected_columns}")
        # print(type(selected_columns))
        print("\n")

# rename to sbipf
def sbipf(reference, target_margins, feature_groups, ipf_algo=ipfn_wrapper, ipf_args={}, n=None,
           epsilon=1e-20): # conv_rate=1e-6, tol=1e-8, verbose = False,
    """ Performs IPF sequentially
    columns_seq: sequence of variables, a list of list
    """
    D_synth = reference.copy() # synthesized is same as the source before IPF starts
    n = reference.shape[0] if n is None else n

    # Shuffle group ordering
    # var_groups = feature_groups.copy()
    # random.shuffle(var_groups)
    var_groups = feature_groups # use the provided groups, no shuffling, and instead use the optimized_bipf() function to search for the best group ordering

    for i in range(len(var_groups)):
        columns = var_groups[i]
        # print(f"\nProcessing group {i+1}/{len(columns_seq)}: {columns}")
        # target_table = c_table(target, target_vars=columns) # target table
        # target_margins = compute_marginals(target_table) # True/Population margins
        # seed_table = seed_table/seed_table.sum() # normalize table
        # selected_target_margins = [target_margins[key] for key in columns if key in target_margins]
        if ipf_algo.__name__ == 'ipfn_wrapper':
            selected_target_margins = [target_margins[key] for key in columns if key in target_margins]
            # Prepare the contingency tables and target marginals or (population) data 
            # seed_table = c_table(reference, target_vars=columns) 
            seed_table = c_table(D_synth, target_vars=columns)
            # Run IPF
            result = ipf_algo(seed_table, selected_target_margins, **ipf_args) # convergence_rate=conv_rate, rate_tolerance=tol,

            # Synthesize
            # w = (result/result.sum())/(seed_table + epsilon)
            w_mat = (result/result.sum())/(seed_table + epsilon)
            # w = w * cur_w
            D_synth = get_samples(D_synth, columns, w_mat, n)

        elif ipf_algo.__name__ == 'ipu_syn':
            selected_target_margins = {key:target_margins[key] for key in columns if key in target_margins}
            D_synth, _ = ipu_syn(primary_seed=D_synth, primary_targets=selected_target_margins, **ipf_args)

    return D_synth



def sbipf_old(reference, target_margins, feature_groups, ipf_algo, runs=1, ipf_args={}, n=None,
          conv_rate=1e-6, tol=1e-8, verbose = False, epsilon=1e-20):
    """ Performs IPF sequentially
    columns_seq: sequence of variables, a list of list
    """
    D_synth = reference.copy() # synthesized is same as the source before IPF starts
    n = reference.shape[0] if n is None else n
    w = 1

    # Shuffle group ordering
    var_groups = feature_groups.copy()
    random.shuffle(var_groups)

    for j in range(runs):
        # old_margins = {col: D_synth[col].value_counts(normalize=True) 
                    #   for col in D_synth.columns}
        
        for i in range(len(var_groups)):
            columns = var_groups[i]
            # print(f"\nProcessing group {i+1}/{len(columns_seq)}: {columns}")
            
            # Prepare the contingency tables and target marginals or (population) data 
            # seed_table = c_table(reference, target_vars=columns) 
            seed_table = c_table(D_synth, target_vars=columns)
            # seed_table = seed_table/seed_table.sum() # normalize table
            selected_target_margins = [target_margins[key] for key in columns if key in target_margins]
            
            # Run IPF
            result = ipf_algo(seed_table, selected_target_margins, convergence_rate=conv_rate, rate_tolerance=tol, **ipf_args) 

            # Synthesize
            # w = (result/result.sum())/(seed_table + epsilon)
            w_mat = (result/result.sum())/(seed_table + epsilon)
            # w = w * cur_w
            D_synth = get_samples(D_synth, columns, w_mat, n)
            #-- take into account previous weights
            # w = w * get_sampling_weights(reference, columns, w_mat)
            # current_weights = get_sampling_weights(D_synth, columns, w_mat)
            # Apply l1 regularization to smooth the weights
            # current_weights /= (1+0.1*np.abs(current_weights))
            # w = w * (n/np.sum(w))
            # print(w)
            # w = pd.Series(1, index=synthesized.index)
            # D_synth = D_synth.sample(n=n, replace=True, weights=current_weights).reset_index(drop=True) 

    # Check convergence
    # new_margins = {col: D_synth[col].value_counts(normalize=True) 
    #                 for col in D_synth.columns}
    
    # max_diff = max(
    #     abs(old_margins[col] - new_margins[col]).max()
    #     for col in D_synth.columns
    # )
    return D_synth


# Implementation of naive blockwise IPF
def nbipf(reference, target_margins, feature_groups, ipf_algo, ipf_args={},
          conv_rate=1e-6, tol=1e-10, n=None, runs=1, epsilon=1e-20):
    """
    Performs Naive Blockwise Iterative Proportional Fitting (IPF).

    Parameters:
    - reference: pd.DataFrame
        The reference DataFrame to start with.
    - target_margins: dict
        Dictionary containing target margins for each variable.
    - feature_groups: list of lists
        Sequence of variable groups (e.g., [['G1'], ['G2'], ...]).
    - ipf_algo: function
        The IPF algorithm to apply.
    - ipf_args: dict
        Additional arguments for the IPF algorithm.
    - n: int, optional
        Number of samples to draw. Defaults to the number of rows in reference.
    - epsilon: float
        Small constant to prevent division by zero.

    Returns:
    - synthesized: pd.DataFrame
        The synthesized DataFrame after sequential IPF.
    - Sampling weights vector
    """
    # Initialize synthesized DataFrame as a copy of the reference
    reference = reference.copy() # To prevent modifying the outside data passed to the function
                                 # What a nightmare!!!
    # synthesized = reference.copy()
    n = reference.shape[0] if n is None else n

    # Initialize combined weights vector as a Series with all ones
    w = pd.Series(1.0, index=reference.index)

    for j in range(runs):
        for i, columns in enumerate(feature_groups):
            # print(f"Processing group {i+1}/{len(columns_seq)}: {columns}")
        
            # Prepare the contingency table for the current group of variables
            seed_table = c_table(reference, target_vars=columns) 
            
            # Extract target margins for the current group
            selected_target_margins = [target_margins[key] for key in columns if key in target_margins]
            
            # Validate that all columns have corresponding target margins
            if len(selected_target_margins) != len(columns):
                missing = set(columns) - set(target_margins.keys())
                raise KeyError(f"Missing target margins for variables: {missing}")
            
            # Run the IPF algorithm to obtain adjusted margins
            result = ipf_algo(seed_table, selected_target_margins, convergence_rate=conv_rate,
                              rate_tolerance=tol, **ipf_args) 

            # Compute the weight adjustment matrix
            w_mat = (result/result.sum()) / (seed_table + epsilon) #TODO correct for bias for the epsilon

            # Continuously reweight source/refernece data 
            # synthesized = get_samples(synthesized, columns, w_mat, n)
        
            # Update weights based on the current group
            # updated_weights = get_sampling_weights(synthesized, columns, w_mat)
            current_weights = get_sampling_weights(reference, columns, w_mat)

            # Check for NaNs or infinite values in updated_weights
            if current_weights.isnull().any() or np.isinf(current_weights).any():
                raise ValueError("Encountered NaN or infinite values in current weights.")
            # normalize weights within each group to avoid scaling issues:
            # current_weights /= current_weights.sum() # approximately no change in results

            # Update cumulative weights
            # Apply l1 regularization to smooth the weights
            # current_weights /= (1+0.01*np.abs(current_weights)) # Helps in some situations with highly correlated variables across groups
            # current_weights /= (1+0.00*(current_weights**2)) # Ridge penalty
            w = w * current_weights
            # Apply l1 regularization to smooth the weights
            # w /= (1+0.01*np.abs(w))
            # w = w * (n / w.sum())

        # Normalize weights to maintain the desired sample size
        total_weight = w.sum()
        if total_weight == 0:
            raise ValueError("Total weight is zero after updating weights.")
        # Take the geometric mean
        # w = w**(1/len(columns_seq))
        w = w * (n / total_weight) # No change in weights if total_weight == n. What if we downsample or upsample?
        
        # print(f"Total weight after normalization: {w.sum()}")
        synthesized = reference.sample(n, replace=True, weights=w)
    return synthesized, w


# A function to find the optimal ordering of groups for blockwise IPF
from itertools import permutations
def optimal_bipf(bipf_func, reference, target_margins, feature_groups, ipf_algo, ipf_args={},
                   n=None, epsilon=1e-20): # conv_rate=1e-6, tol=1e-8,
    """
    Finds the optimal ordering of groups for Naive Blockwise IPF.

    Parameters:
    - nbipf_func: function
        The Naive Blockwise IPF function to apply.
    - For the rest of the parameters, see the docstring of nbipf_func.
    Returns:
    - best_ordering: list
        The optimal ordering of feature groups.
    """

    best_ordering = None
    best_result = None

    for perm in permutations(feature_groups):
        # convert perm to a list of lists
        perm = list(perm)

        if bipf_func.__name__ == 'sbipf':
            # set weight vector to None
            w = None
            synthesized = sbipf(reference, target_margins, perm, ipf_algo, ipf_args=ipf_args,
                                n=n, epsilon=epsilon)
        elif bipf_func.__name__ == 'nbipf':
            synthesized, w = nbipf(reference, target_margins, perm, ipf_algo, ipf_args=ipf_args,
                                 n=n, epsilon=epsilon)
        else:
            raise ValueError("Unsupported bipf_func. Use 'sbipf' or 'nbipf'.")
        
        # Evaluate the result (e.g., using some metric)
        # Here we can use a simple metric like the sum of absolute differences from target margins
        print(type(synthesized))
        print(synthesized.shape)
        result_metric = sum(abs(synthesized[col].value_counts(normalize=False) - target_margins[col]).sum() for col in synthesized.columns)

        if best_result is None or result_metric < best_result:
            best_result = result_metric
            best_ordering = perm
        # print(best_result)

    return synthesized, best_ordering, w




def seq_ipf2b(reference, target_margins, columns_seq, ipf_algo, ipf_args={}, n=None, key_columns=None, epsilon=1e-20):
    """
    Performs Sequential Iterative Proportional Fitting (IPF).

    Parameters:
    - reference: pd.DataFrame
        The reference DataFrame to start with.
    - target_margins: dict
        Dictionary containing target margins for each variable.
    - columns_seq: list of lists
        Sequence of variable groups (e.g., [['G1'], ['G2'], ...]).
    - ipf_algo: function
        The IPF algorithm to apply.
    - ipf_args: dict
        Additional arguments for the IPF algorithm.
    - n: int, optional
        Number of samples to draw. Defaults to the number of rows in reference.
    - epsilon: float
        Small constant to prevent division by zero.

    Returns:
    - synthesized: pd.DataFrame
        The synthesized DataFrame after sequential IPF.
    - Sampling weights vector
    """
    # Initialize synthesized DataFrame as a copy of the reference
    reference = reference.copy() # To prevent modifying the outside data passed to the function
                                 # What a nightmare!!!
    synthesized = reference.copy()
    n = reference.shape[0] if n is None else n
    if key_columns is None:
          # Set columns in first group (default core group) as key columns
          key_columns = columns_seq[0]
    #     # Add the ID column starting from 1 up to the number of rows
    #     reference['id'] = range(1, len(reference) + 1)
    #     # Set the key column to "id"
    #     key_column = "id"

    # Initialize combined weights vector as a Series with all ones
    # w = pd.Series(1.0, index=synthesized.index)

    # Ensure the key column is in the DataFrame
    # if key_column not in reference.columns:
    #     raise ValueError(f"The key column '{key_column}' is not in the reference DataFrame.")

    # Create sub-DataFrames
    sub_dataframes = {}
    # Start with an empty DataFrame containing just the key/id column
    # print(len(reference))
    # merged_df = pd.DataFrame({key_column: range(1, len(reference) + 1)})

    for i, columns in enumerate(columns_seq):
        print(f"Processing group {i+1}/{len(columns_seq)}: {columns}")
    
        # Prepare the contingency table for the current group of variables
        seed_table = c_table(reference, target_vars=columns) 
        
        # Extract target margins for the current group
        selected_target_margins = [target_margins[key] for key in columns if key in target_margins]
        
        # Validate that all columns have corresponding target margins
        if len(selected_target_margins) != len(columns):
            missing = set(columns) - set(target_margins.keys())
            raise KeyError(f"Missing target margins for variables: {missing}")
        
        # Run the IPF algorithm to obtain adjusted margins
        result = ipf_algo(seed_table, selected_target_margins, **ipf_args) 

        # Compute the weight adjustment matrix
        w_mat = result / (seed_table + epsilon) #TODO correct for bias for the epsilon

        ## Continuously reweight source/refernece data 
        # synthesized = get_samples(synthesized, columns, w_mat, n)
        
    
        # Update weights based on the current group
        # updated_weights = get_sampling_weights(synthesized, columns, w_mat)
        # current_weights = get_sampling_weights(reference, columns, w_mat)

        # Check for NaNs or infinite values in updated_weights
        # if current_weights.isnull().any() or np.isinf(current_weights).any():
        #     raise ValueError("Encountered NaN or infinite values in current weights.")
        # normalize weights within each group to avoid scaling issues:
        # current_weights /= current_weights.sum()

        # Update cumulative weights
        # w = w * current_weights

        # Sample the synthesized DataFrame based on updated weights
        # To maintain alignment, we use the 'weights' parameter correctly
        # synthesized = synthesized.sample(n=n, replace=True, weights=w, random_state=42).reset_index(drop=True)
        
        # Update the weights to reflect the resampling
        # Each sampled row inherits the weight from its source row
        # Create a new weights Series based on the sampled indices
        # Since we've reset the index, we need to align weights accordingly
        # We'll create a new weights Series by taking weights of sampled rows
        # Note: 'sampled_indices' retains the original indices before reset
        # sampled_indices = synthesized.index  # After reset, indices are 0 to n-1
        # To map back to original weights before reset, we need to capture them before resetting
        # Adjusting the approach:
        
        # Instead of resetting the index immediately, capture the weights first
        # sampled = synthesized.sample(n=n, replace=True, weights=w) # , random_state=42
        # sampled_w = w.loc[sampled.index].copy()
        
        # # Reset index to ensure alignment
        # sampled = sampled.reset_index(drop=True)
        # sampled_w = sampled_w.reset_index(drop=True)
        
        # Assign updated weights to the sampled DataFrame
        # w = sampled_w
        # print(w)
        # synthesized = sampled

        # Debug: Check for NaNs in weights
        # if w.isnull().any():
        #     print("Warning: NaN values detected in weights after sampling.")
        #     w = w.fillna(0)  # Or handle appropriately
        # if np.isinf(w).any():
        #     print("Warning: Infinite values detected in weights after sampling.")
        #     w = w.replace([np.inf, -np.inf], 0)

        
        # sub_dataframes[f"group_{i}"]  = get_samples(reference[columns_to_include], columns, w_mat, n)
        # Synthesize data for the kth group
        if i == 0: # Assume first group is the anchor group
            base_df = get_samples(reference, columns, w_mat, n)
            dfk = base_df[columns] # core group synthesized data
        else:
            # Generate data for (k-1)th group variables
            columns_to_include = [key_columns] + columns
            dfk = get_samples(reference[columns_to_include], columns, w_mat, n)
        sub_dataframes[f"group_{i}"] = dfk
    # Normalize weights to maintain the desired sample size
    # total_weight = w.sum()
    # if total_weight == 0:
    #     raise ValueError("Total weight is zero after updating weights.")
    # w = w * (n / total_weight)
    
    print(f"Total weight after normalization: {w.sum()}")
    
    return synthesized, sub_dataframes


def seq_ipf2c(reference, target_margins, columns_seq, ipf_algo, ipf_args={}, n=None, epsilon=1e-10):
    """
    Performs Sequential Iterative Proportional Fitting (IPF) on subsets of variables.

    Parameters:
    - reference: pd.DataFrame
        The reference DataFrame to start with.
    - target_margins: dict
        Dictionary containing target margins for each variable.
    - columns_seq: list of lists
        Sequence of variable groups (e.g., [['G1'], ['G2'], ...]).
    - ipf_algo: function
        The IPF algorithm to apply.
    - ipf_args: dict
        Additional arguments for the IPF algorithm.
    - n: int, optional
        Number of samples to draw. Defaults to the number of rows in reference.
    - epsilon: float
        Small constant to prevent division by zero.

    Returns:
    - synthesized: pd.DataFrame
        The synthesized DataFrame after sequential IPF.
    """
    # Initialize synthesized DataFrame as a copy of the reference
    synthesized = reference.copy()
    n = reference.shape[0] if n is None else n

    # Initialize weights as a separate column
    synthesized['weight'] = 1.0

    for i, columns in enumerate(columns_seq):
        print(f"\nProcessing group {i+1}/{len(columns_seq)}: {columns}")

        # Prepare the contingency table for the current group of variables
        seed_table = c_table(reference, target_vars=columns) 
        
        # Extract target margins for the current group
        selected_target_margins = [target_margins[key] for key in columns if key in target_margins]
        
        # Validate that all columns have corresponding target margins
        if len(selected_target_margins) != len(columns):
            missing = set(columns) - set(target_margins.keys())
            raise KeyError(f"Missing target margins for variables: {missing}")
        
        # Run the IPF algorithm to obtain adjusted margins
        result = ipf_algo(seed_table, selected_target_margins, **ipf_args) 

        # Compute the weight adjustment matrix
        w_mat = result / (seed_table + epsilon)
        
        # Update weights based on the current group
        updated_weights = get_sampling_weights(synthesized, columns, w_mat)

        # Ensure that updated_weights does not contain NaN or infinite values
        if updated_weights.isnull().any() or np.isinf(updated_weights).any():
            raise ValueError("Encountered NaN or infinite values in updated weights.")
        
        # Update the 'weight' column with cumulative weights
        synthesized['weight'] = synthesized['weight'] * updated_weights
        
        # Normalize weights to maintain the desired sample size
        total_weight = synthesized['weight'].sum()
        if total_weight == 0:
            raise ValueError("Total weight is zero after updating weights.")
        synthesized['weight'] = synthesized['weight'] * (n / total_weight)
        
        print(f"Total weight after normalization: {synthesized['weight'].sum()}")

        # Sample the synthesized DataFrame based on updated weights
        synthesized = synthesized.sample(n=n, replace=True, weights='weight').reset_index(drop=True)

        # After sampling, reset weights to reflect the sampling
        # Each sampled row inherits the weight from its source row
        # Since we've reset the index, align weights accordingly
        # Here, we assume that 'get_sampling_weights' accounts for the sampling process
        # Therefore, we can keep the 'weight' column as is for the next iteration

        # Optional: Handle any NaNs that might have arisen during sampling
        if synthesized['weight'].isnull().any():
            print("Warning: NaN values detected in weights after sampling. Filling with zeros.")
            synthesized['weight'] = synthesized['weight'].fillna(0)
        if np.isinf(synthesized['weight']).any():
            print("Warning: Infinite values detected in weights after sampling. Replacing with zeros.")
            synthesized['weight'] = synthesized['weight'].replace([np.inf, -np.inf], 0)

    # Optionally, drop the 'weight' column if it's no longer needed
    # synthesized = synthesized.drop(columns=['weight'])

    return synthesized

# Based on Predictive Join Synthesis (PJS) algorithm
# Adapted from SynC: A Copula based Framework for Generating Synthetic Data from Aggregated Sources:
#  https://ieeexplore.ieee.org/abstract/document/9346329
# Let's try to implement this algorithm with minimal features for the anchor group, say 2 features
def seq_ipf3(reference, target_margins, columns_seq, key_columns, ipf_algo, ipf_args={}, n=None, epsilon=1e-20):
    """
    Performs Sequential Iterative Proportional Fitting (IPF) with PJS.

    Parameters:
    - reference: pd.DataFrame
        The reference DataFrame to start with.
    - target_margins: dict
        Dictionary containing target margins for each variable.
    - columns_seq: list of lists
        Sequence of variable groups (e.g., [['G1'], ['G2'], ...]).
    - ipf_algo: function
        The IPF algorithm to apply.
    - ipf_args: dict
        Additional arguments for the IPF algorithm.
    - n: int, optional
        Number of samples to draw. Defaults to the number of rows in reference.
    - epsilon: float
        Small constant to prevent division by zero.

    Returns:
    - synthesized: pd.DataFrame
        The synthesized DataFrame after sequential IPF.
    - Sampling weights vector
    """
    # Initialize synthesized DataFrame as a copy of the reference
    reference = reference.copy() # To prevent modifying the outside data passed to the function
                                 # What a nightmare!!!
    n = reference.shape[0] if n is None else n

    # Create sub-DataFrames
    group_dataframes = {}
    

    for i, columns in enumerate(columns_seq):
        print(f"Processing group {i+1}/{len(columns_seq)}: {columns}")

        if i == 0: # Assume first group is the anchor group
            columns_to_include = key_columns
        else:
            # Generate data for (k-1)th group variables
            columns_to_include = key_columns + columns
            
        # Prepare the contingency table for the current group of variables
        seed_table = c_table(reference, target_vars=columns_to_include) 
        # print(seed_table.shape)
        # Extract target margins for the current group
        selected_target_margins = [target_margins[key] for key in columns_to_include if key in target_margins]
        # print(selected_target_margins)
        
        # Validate that all columns have corresponding target margins
        if len(selected_target_margins) != len(columns_to_include):
            missing = set(columns) - set(target_margins.keys())
            raise KeyError(f"Missing target margins for variables: {missing}")
        
        # Run the IPF algorithm to obtain adjusted margins
        result = ipf_algo(seed_table, selected_target_margins, **ipf_args) 

        # Compute the weight adjustment matrix
        w_mat = result / (seed_table + epsilon) #TODO correct for bias for the epsilon

        # Synthesize data for the kth group
        group_dataframes[f"group_{i}"] = get_samples(reference[columns_to_include], columns_to_include, w_mat, n)
        
    
    return group_dataframes


##--- References
# Iterative Proportional Fitting, Two Dimensions: https://datascience.oneoffcoder.com/ipf.html
# Iterative Proportional Fitting, Higher Dimensions: https://datascience.oneoffcoder.com/ipf-ii.html
# - (check out) Iterative Proportional Fitting Information, Code, and Links: https://edyhsgr.github.io/datafitting.html 
# - (check out) Python IPFn library: https://github.com/Dirguis/ipfn/tree/master 


#--------- Evaluation Metrics ---------

## Metrics for comparing reference and fitted contingency tables

from scipy.special import rel_entr
from scipy.special import kl_div
from scipy.stats import entropy
from scipy.stats import wasserstein_distance

# compute_dependence_measures()
def chisquared_result(data, selected_features=None):
    results = {}
    if selected_features is None:
        selected_features = list(data.columns)

    for model in data:
        X = c_table(data[model], selected_features)
        chi_stat, cramer_v = compute_dependence_measures(X)
        results[model] = [chi_stat, cramer_v]
    return results

# compute crossproduct ratios for structure conservation
def cross_product_ratios(table):
    # look for better options for handling empty cells
    rows, cols = table.shape
    ratios = np.zeros((rows - 1, cols - 1))
    
    for i in range(rows - 1):
        for j in range(cols - 1):
            a = table[i, j]
            b = table[i, j + 1]
            c = table[i + 1, j]
            d = table[i + 1, j + 1]
            if b * c != 0:
                ratios[i, j] = (a * d) / (b * c)
            else:
                ratios[i, j] = np.nan  # Avoid division by zero
    
    return ratios


def compute_2x2_odds_ratio(subtable):
    """
    Compute the odds ratio for a 2x2 subtable.
    """
    a = subtable[0, 0]
    b = subtable[0, 1]
    c = subtable[1, 0]
    d = subtable[1, 1]
    if b * c != 0:
        return (a * d) / (b * c)
    else:
        return np.nan  # Avoid division by zero

def cross_product_ratios_nd(table):
    """
    Compute the cross-product ratios for an n-dimensional contingency table.
    """
    shape = table.shape
    ratios = np.full([dim - 1 for dim in shape], np.nan)
    
    # Iterate over all possible 2x2 sub-tables
    it = np.nditer(ratios, flags=['multi_index'])
    while not it.finished:
        index = it.multi_index
        slices = tuple(slice(i, i + 2) for i in index)
        subtable = table[slices]
        
        # Ensure the subtable is 2x2 in all dimensions
        if subtable.shape == (2, 2, *subtable.shape[2:]):
            # Flatten the first two dimensions to apply the 2x2 odds ratio
            subtable_2x2 = subtable.reshape(2, 2, -1)
            ratios[index] = np.nanmean([compute_2x2_odds_ratio(subtable_2x2[:, :, k]) for k in range(subtable_2x2.shape[2])])
        
        it.iternext()
    
    return ratios

# Example usage for 3-dimensional contingency tables
# reference_table_3d = np.array([
#     [[10, 5], [20, 15]],
#     [[30, 25], [40, 35]]
# ])
# cross_product_ratios_nd(reference_table_3d)

def kld(p, q):
    p = p/p.sum()
    q = q/q.sum()
    return np.sum(rel_entr(p, q))

def compare_cont_tables(ref_table, fitted_table):
    # normalize the tables
    ref_table_norm = ref_table/ref_table.sum()
    fitted_table_norm = fitted_table/fitted_table.sum()
    
    
    # kl_divergence = np.sum(rel_entr(fitted_table_norm, ref_table_norm))
    kl_divergence = np.sum(kl_div(fitted_table_norm, ref_table_norm)) # no significant difference
    
    # Total Variation Distance
    total_variation_distance = 0.5 * np.sum(np.abs(ref_table_norm - fitted_table_norm))
    
    # Hellinger Distance
    sqrt_ref = np.sqrt(ref_table_norm)
    sqrt_fit = np.sqrt(fitted_table_norm)
    hellinger_distance = np.sqrt(0.5 * np.sum((sqrt_ref - sqrt_fit) ** 2))
    
    # Wasserstein Distance
    w_dist = wasserstein_distance(ref_table_norm.flatten(),
                                  fitted_table_norm.flatten())
    
    return {"KL-divergence":kl_divergence,
            "Hellinger dist": hellinger_distance,
            "TVD": total_variation_distance,
            "W_distance": w_dist}


## Metrics for comparing synthetic and reference data distributions

#----- Marginal fit metrics
from scipy.stats import entropy

def compare_margins(reference_margins, target_margins, metric='KLD', epsilon=0):
    result = 0
    for i in range(len(reference_margins)):
        # Calculate KL divergence (adding a small value to avoid log(0))
        kl_divergence = entropy(reference_margins[i] + epsilon, target_margins[i])
        result += kl_divergence
        
    return result


#----- Standardized Root Mean Squared Error (SRMSE)
# functions taken from: https://github.com/PascalJD/copulapopgen/blob/main/utils.py
def srmse(data1, data2):
    """ Compute Standardized Root Mean Squared Error between two datasets.
    
    data1: reference data
    data2: synthetic data
    
    Reference: https://www.researchgate.net/publication/282815687_A_Bayesian_network_approach_for_population_synthesis
    """
    columns = list(data1.columns.values)
    # Relative frequency
    data1_f = data1.value_counts(normalize=True)
    data2_f = data2.value_counts(normalize=True)
    
    # Total numbers of categories
    Mi = [data1_f.index.get_level_values(l).union(data2_f.index.get_level_values(l)).unique().size for l in range(len(columns))]
    M = np.prod(Mi)
    # SRMSE
    SRMSE = ((data1_f.subtract(data2_f, fill_value=0)**2) * M).sum()**(.5)
    return SRMSE

def project_srmse(data1, data2, columns, max_projection=5):
    """ Prject SRMSE over subsets of n variables, with n ranging from 1 to max_projection.
    
    max_projection: subsets of n variables
    """
    srmse_dict = {}
    for i in range(1, max_projection+1):
        tuples = list(itertools.combinations(columns, i))  # No repeated elements
        SRMSE = 0 
        for tuple in tuples:
            SRMSE += srmse(
                data1.drop(list(columns.difference(tuple)), axis=1),
                data2.drop(list(columns.difference(tuple)), axis=1)
            )
        SRMSE /= len(tuples)
        srmse_dict["SRMSE "+str(i)] = np.round(SRMSE, 3)
    return srmse_dict

def sampling_zeros(source, target, synthetic):
    """Count the combinations of variables from the synthetic data which 
    are in the test set but not in the training set.

    Reference: https://arxiv.org/pdf/1909.07689.pdf
    """
    source_set = set(tuple(i) for i in source.to_numpy())
    target_set = set(tuple(i) for i in target.to_numpy())
    synthetic_set = set(tuple(i) for i in synthetic.to_numpy())
    zeros = synthetic_set.intersection(target_set) - source_set
    return len(zeros)


def result_table(target, synthetic_data, columns, max_projection=5, save=False, path=""):
    # Calculate SRMSE and zeros
    srmse_dict = {}
    for model in synthetic_data:
        srmse_dict[model] = {}
        df = synthetic_data[model] 
        for i in range(1, max_projection+1):
            tuples = list(itertools.combinations(columns, i))  # No repeated elements
            SRMSE = 0 
            for tuple in tuples:
                SRMSE += srmse(
                    target.drop(list(columns.difference(tuple)), axis=1),
                    df.drop(list(columns.difference(tuple)), axis=1)
                )
            SRMSE /= len(tuples)
            srmse_dict[model]["SRMSE "+str(i)] = SRMSE
        # srmse_dict[model]["Zeros"] = sampling_zeros(source, target, df)
    
    table = [] 
    for model in srmse_dict:
        table.append(
            pd.DataFrame({i:srmse_dict[model][i] for i in srmse_dict[model]}, index=[model]))
    table = pd.concat(table)

    if save: 
        table.to_csv(f"{path}/result_table.csv")
        
    return table
# Codes for the paper “Scalable Population Synthesis with Deep Generative Modeling”: https://github.com/stasmix/popsynth
