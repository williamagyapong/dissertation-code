#----------------------------------------------------------
# Description: Core implementation codes for the IPF algorithm
# Author(s): William O. Agyapong
# Created on: 06-13-2024
# Date Last Modified: 04-04-2025
#----------------------------------------------------------

import numpy as np
import pandas as pd
import itertools
import functools


#----- Standard IPF: 
# Credit to the Data Science Topics website for providing the Python
# implementation codes adapted from: https://datascience.oneoffcoder.com/ipf-ii.html

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


#------ A wrapper function for ipfn
from ipfn import ipfn
def ipfn_wrapper(X, target_margins, convergence_rate=1e-6, rate_tolerance = 1e-8, other_args={}):
    # other_args={}
    dimensions = [[i] for i in range(len(target_margins))]
    # IPF = ipfn.ipfn(X, target_margins , dimensions, **other_args) # , convergence_rate=1e-10
    IPF = ipfn.ipfn(X, target_margins , dimensions, convergence_rate=convergence_rate, rate_tolerance=rate_tolerance, **other_args)
    m = IPF.iteration()
    return m
# Standard IPF based on ipfn
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
    
    # Run the standard IPF on the data
    X_ref_adjusted = ipfn_wrapper(X_ref, target_margins, convergence_rate=conv_rate, rate_tolerance=tol, other_args=ipf_args) # , other_args={"max_iteration":10000}
    # X_ref_adjusted = ipf(X_ref, target_margins) # convergence issues with zero margins

    # Resampling (synthesizing) data
    w = (X_ref_adjusted/X_ref_adjusted.sum())/(X_ref+1e-20) # 
    syn_ref_set = get_samples(ref_set, selected_features, w, n=target_set.shape[0])
    
    return syn_ref_set, X_ref_adjusted


## Generate sampling weights for each sample in the source data
# This function distributes the weights from the adjusted contingency table to the samples/respondents
def get_sampling_weights(df, f, w):
    weights =  {k: v for k, v in zip(list(itertools.product(*[sorted(df[c].unique()) for c in f])), np.ravel(w))}
    w = df.apply(lambda r: weights[tuple([r[c] for c in f])], axis=1)
    return w
    
def get_samples(df, f, w_mat, n=10_000):
    """
    w_mat: is the weight matrix/table based on the adjusted contingency table
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
    #TODO validate input, length of weights
    n = n if n is not None else int(np.round(weights.sum(),0))
    return df.sample(n=n, replace=True, weights=weights) 

# Generate IPF-based synthetic/reweighted data
def std_ipf_syn(df, X, X_adjusted, selected_features=None, epsilon=1e-10):
    # epsilon: a small number that aleviates division by zero problems
    if selected_features is None:
        selected_features = list(df.columns)

    # w = (X_adjusted/X_adjusted.sum())/(X+epsilon) # 
    w = (X_adjusted)/(X+epsilon) # 
    sampled_df = get_samples(df, selected_features, w, n=df.shape[0])
    #TODO Consider attaching sampling weights to the data
    return sampled_df 


#---------- Blockwise IPF ----------
# This is where the implementation of the blockwise IPF starts

import random 


def group_features(selected_features, n_group=2, random_seed=None):
    """Group features randomly into n_group groups
    """
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

def group_features_dep(df):
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

# This function performs sequential blockwise IPF
def sbipf(reference, target_margins, var_groups, ipf_algo, runs=1, ipf_args={}, n=None, verbose = False, epsilon=1e-20):
    """ Performs IPF sequentially
    columns_seq: sequence of variables, a list of list
    """
    D_synth = reference.copy() # synthesized is same as the source before IPF starts
    n = reference.shape[0] if n is None else n
    w = 1

    # Shuffle group ordering
    var_groups = var_groups.copy()
    random.shuffle(var_groups)

    for j in range(runs):
        for i in range(len(var_groups)):
            columns = var_groups[i]
            # print(f"\nProcessing group {i+1}/{len(columns_seq)}: {columns}")
            
            # Prepare the contingency tables and target marginals or (population) data 
            # seed_table = c_table(reference, target_vars=columns) 
            seed_table = c_table(D_synth, target_vars=columns)
            # seed_table = seed_table/seed_table.sum() # normalize table
            selected_target_margins = [target_margins[key] for key in columns if key in target_margins]
            
            # Run IPF
            result = ipf_algo(seed_table, selected_target_margins, **ipf_args) 

            # Synthesize
            # w = (result/result.sum())/(seed_table + epsilon)
            w_mat = (result)/(seed_table + epsilon)
            # w = w * cur_w
            D_synth = get_samples(D_synth, columns, w_mat, n)
            
    return D_synth


# Implementation of naive blockwise IPF
def nbipf(reference, target_margins, columns_seq, ipf_algo, ipf_args={}, n=None, runs=1, epsilon=1e-20):
    """
    Performs Naive Blockwise Iterative Proportional Fitting (IPF).

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
    # synthesized = reference.copy()
    n = reference.shape[0] if n is None else n

    # Initialize combined weights vector as a Series with all ones
    w = pd.Series(1.0, index=reference.index)

    for j in range(runs):
        for i, columns in enumerate(columns_seq):
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
            result = ipf_algo(seed_table, selected_target_margins, **ipf_args) 

            # Compute the weight adjustment matrix
            w_mat = result / (seed_table + epsilon) #TODO correct for bias for the epsilon
        
            # Update weights based on the current group
            # updated_weights = get_sampling_weights(synthesized, columns, w_mat)
            current_weights = get_sampling_weights(reference, columns, w_mat)

            # Check for NaNs or infinite values in updated_weights
            if current_weights.isnull().any() or np.isinf(current_weights).any():
                raise ValueError("Encountered NaN or infinite values in current weights.")
            # normalize weights within each group to avoid scaling issues:
            # current_weights /= current_weights.sum() # approximately no change in results

            # Update cumulative weights
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
        w = w * (n / total_weight)
        
        # print(f"Total weight after normalization: {w.sum()}")
        synthesized = reference.sample(n, replace=True, weights=w)
    return synthesized, w



# This is a proposed third implementation of BIPF based on Predictive Join Synthesis (PJS) algorithm
# Adapted from SynC: A Copula based Framework for Generating Synthetic Data from Aggregated Sources:
#  https://ieeexplore.ieee.org/abstract/document/9346329
# Let's try to implement this algorithm with minimal features for the anchor group, say 2 features
# Under development, and will be included in the dissertation as one of the future works
def bipf_pjs(reference, target_margins, columns_seq, key_columns, ipf_algo, ipf_args={}, n=None, epsilon=1e-20):
    """
    Performs Blockwise Iterative Proportional Fitting (IPF) with Predictive Join Synthesis (PJS).
    The idea is to train a predictive model to predict features in other groups using a selected group (called the anchor or core group) as a feature space.

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
        The synthesized DataFrame after applying predictive matching.
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

