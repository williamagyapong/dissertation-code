#----------------------------------------------------------
# Description: Houses more general utility functions for the project
# Author(s): William O. Agyapong
# Created on: 05-11-2024
# Date Last Modified: 04/08/2025
#----------------------------------------------------------

import numpy as np
import pandas as pd
import itertools
import subprocess
import scipy.stats as stats
from scipy.stats import entropy
from scipy.stats import norm


def set_seed(seed=123):
    # Set a random seed for NumPy
    np.random.seed(seed)
    # Set a random seed for Python's built-in random module
    import random
    random.seed(seed)
    return None 

# -------- Data Simulation Functions ----------------
def generate_groupings(p, exclude_indices=None):
    """
    Generate groupings of variables based on the number of variables p.
    The groupings are based on the divisors of p, excluding 1 and p itself. 
    Args:
        p: Number of variables.
        exclude_indices (list of int): Indices of sublists to exclude.
    Returns:
        A list of lists, where each inner list represents a grouping of variables.
    """
    # Get all divisors of p that are >= 2 and < p
    divisors = [i for i in range(2, p) if p % i == 0]
    
    # Generate list of groupings based on divisors
    groupings = []
    for divisor in divisors:
        num_groups = p // divisor  # Number of groups
        group_size = divisor  # Size of each group
        grouping = [group_size] * num_groups
        groupings.append(grouping)
    if exclude_indices is not None:
        groupings = [sublist for i, sublist in enumerate(groupings) if i not in exclude_indices]
    return groupings 

def simulate_independent_groups(num_samples, group_sizes, categories_per_group=None, rho=0.8, random_seed=None):
    """
    Simulate categorical data such that variables within a group are correlated but are independent of other groups.
    This version allows varying numbers of categories per group.
    
    Args:
        num_samples: Number of samples (rows) in the dataset.
        num_groups: Number of independent groups.
        group_sizes: A list with the number of variables in each group.
        categories_per_group: A list specifying the number of categories for each group. If None, defaults to 3 categories per group.
        rho: Strength of correlation, a float value (0 to 1), controlling within-group dependency strength.
        
    Returns:
        A pandas DataFrame with simulated dependent categorical data.
    """
    data = pd.DataFrame()
    # np.random.seed(random_seed) # random_seed=123
    if random_seed is not None:
        np.random.seed(random_seed)
    num_groups = len(group_sizes)

    if categories_per_group is None:
        categories_per_group = [2] * num_groups  # Default to 2 categories per group if not specified
    
    for group_idx in range(num_groups):
        group_size = group_sizes[group_idx]
        num_categories = categories_per_group[group_idx]
        
        # Generate a base variable to induce correlation within the group
        base_categories = list(range(num_categories))
        base_variable = np.random.choice(base_categories, size=num_samples)
        
        for var_idx in range(group_size):
            column_name = f"G{group_idx+1}_V{var_idx+1}"
            
            # Map the base variable to the variable's categories to create dependency
            dependent_var = np.random.choice(base_categories, size=num_samples)
            for i, cat in enumerate(base_categories):
                mask = base_variable == cat
                dependent_var[mask] = cat
            
            # Add randomness to reduce perfect correlation
            randomness = np.random.choice(base_categories, size=num_samples)
            dependent_var = np.where(np.random.rand(num_samples) < rho, dependent_var, randomness)
            
            data[column_name] = dependent_var
    
    return data


def simulate_independent_groups_mvn(
    num_samples,  
    group_sizes, 
    categories_per_group=None, 
    rho=0.8,
    random_seed=None
):
    """
    Simulate data using multivariate normal latent variables with independent groups:
    simulate categorical data such that variables within a group are correlated and sampled 
    from a multivariate normal distribution and binned into categories.

    Args:
        num_samples: Number of samples (rows) in the dataset.
        num_groups: Number of independent groups.
        group_sizes: A list with the number of variables in each group.
        categories_per_group: A list specifying the number of categories for each group.
                              If None, defaults to 2 categories per group.
        rho: Correlation strength for within-group dependency (default=0.8).

    Returns:
        A pandas DataFrame with simulated dependent categorical data as well as the latent data.
    """

    # Set seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    data = pd.DataFrame()
    latent_data = pd.DataFrame()
   
    
    num_groups = len(group_sizes)
    
    if categories_per_group is None:
        categories_per_group = [2] * num_groups
    
    for group_idx in range(num_groups):
        group_size = group_sizes[group_idx]
        num_categories = categories_per_group[group_idx]
        
        # Sample a correlation value for the group within a range around the given rho. This introduces some variability across groups.
        group_rho = np.random.uniform(max(rho - 0.1, 0), min(rho + 0.1, 1))
        # Create a covariance matrix for the multivariate normal distribution
        corr_matrix = np.full((group_size, group_size), group_rho)
        np.fill_diagonal(corr_matrix, 1)  # Set diagonal to 1
        cov_matrix = corr_matrix  # Assume unit variances for simplicity
        
        # Sample data from a multivariate normal distribution
        multivariate_data = np.random.multivariate_normal(
            mean=np.zeros(group_size), cov=cov_matrix, size=num_samples
        )
        
        # Bin the continuous data into categories
        for var_idx in range(group_size):
            column_name = f"Group{group_idx+1}_V{var_idx+1}"
            
            # Determine bin edges based on quantiles
            bin_edges = np.linspace(0, 1, num_categories + 1)
            bin_quantiles = norm.ppf(bin_edges)
            
            # Assign categories based on bin edges
            epsilon = np.random.normal(0, 0.1, num_samples) # add a small independent noise term to each latent variable
            binned_data = np.digitize(multivariate_data[:, var_idx]+epsilon, bin_quantiles) - 1
            binned_data[binned_data == num_categories] = num_categories - 1  # Adjust edge case
            
            data[column_name] = binned_data
            latent_data[column_name] = multivariate_data[:, var_idx]
    return data, latent_data

def simulate_relaxed_independent_groups_mvn(
    num_samples, 
    group_sizes, 
    categories_per_group, 
    within_group_rho=0.8, 
    cross_group_rho=0.4,
    random_seed=None
):
    """
    Simulate categorical data using multivariate normal latent variables 
    with both within-group and cross-group correlation.
    
    Args:
        num_samples: Number of samples.
        group_sizes: A list specifying the number of variables in each group.
        categories_per_group: A list specifying the number of categories per group.
        within_group_corr: Correlation strength within each group (default=0.8).
        cross_group_corr: Correlation strength across groups (default=0.3).
        
    Returns:
        A pandas DataFrame with simulated categorical data.
    """
    # Set seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    total_vars = sum(group_sizes)
    
    # Construct full correlation matrix (global covariance structure)
    full_corr_matrix = np.eye(total_vars)  # Start with an identity matrix
    
    start_idx = 0
    for group_idx, group_size in enumerate(group_sizes):
        end_idx = start_idx + group_size
        
        #--- Define within-group correlations
        # Sample a correlation value for the group within a range around the given rho. This introduces some variability across groups.
        group_rho = np.random.uniform(max(within_group_rho - 0.1, 0), min(within_group_rho + 0.1, 1))
        group_corr_block = np.full((group_size, group_size), group_rho)
        np.fill_diagonal(group_corr_block, 1)  # Set diagonal to 1
        
        # Insert within-group correlation block
        full_corr_matrix[start_idx:end_idx, start_idx:end_idx] = group_corr_block
        
        start_idx = end_idx  # Move index
    
    # Introduce cross-group correlations
    for i in range(len(group_sizes)):
        for j in range(i + 1, len(group_sizes)):  # Only fill upper triangular part
            start_i = sum(group_sizes[:i])
            end_i = start_i + group_sizes[i]
            start_j = sum(group_sizes[:j])
            end_j = start_j + group_sizes[j]
            
            cross_corr_block = np.full((group_sizes[i], group_sizes[j]), cross_group_rho)
            full_corr_matrix[start_i:end_i, start_j:end_j] = cross_corr_block
            full_corr_matrix[start_j:end_j, start_i:end_i] = cross_corr_block.T  # Mirror to lower triangular

    # Convert to covariance matrix
    cov_matrix = full_corr_matrix  # Assuming unit variances

    # Generate multivariate normal latent variables
    latent_data = np.random.multivariate_normal(mean=np.zeros(total_vars), cov=cov_matrix, size=num_samples)

    # Convert latent variables into categorical variables
    data = pd.DataFrame()
    start_idx = 0
    for group_idx, group_size in enumerate(group_sizes):
        end_idx = start_idx + group_size
        for var_idx in range(group_size):
            column_name = f"Group{group_idx+1}_Var{var_idx+1}"
            num_categories = categories_per_group[group_idx]
            
            # Define bin edges based on quantiles
            bin_edges = np.linspace(0, 1, num_categories + 1)
            bin_quantiles = norm.ppf(bin_edges)
            
            # Assign categories
            binned_data = np.digitize(latent_data[:, start_idx + var_idx], bin_quantiles) - 1
            binned_data[binned_data == num_categories] = num_categories - 1  # Adjust edge case
            
            data[column_name] = binned_data
        
        start_idx = end_idx  # Move index

    return data, latent_data

def get_variable_names_by_group(variable_names, group_sizes):
    """
    Generate a list of lists containing variable names for each simulated group.
    
    Args:
        data: The simulated dataset (pandas DataFrame).
        group_sizes: A list with the number of variables in each group.
        
    Returns:
        A list of lists containing variable names for each group.
    """
    grouped_variable_names = []
    
    start_idx = 0
    for group_idx, group_size in enumerate(group_sizes):
        end_idx = start_idx + group_size
        group_vars = variable_names[start_idx:end_idx]
        grouped_variable_names.append(group_vars)
        start_idx = end_idx
    
    return grouped_variable_names

def get_variable_indices_by_group(group_sizes):
    """
    Generate a list of lists containing variable indices for each group.
    
    Args:
        num_groups: Number of independent groups.
        group_sizes: A list with the number of variables in each group.
        
    Returns:
        A list of lists containing variable indices for each group.
    """
    grouped_variable_indices = []
    start_idx = 0
    
    for group_size in group_sizes:
        end_idx = start_idx + group_size
        group_indices = list(range(start_idx, end_idx))
        grouped_variable_indices.append(group_indices)
        start_idx = end_idx
    
    return grouped_variable_indices


def ref_target_split(df, low=0.5, high=1.5, split_ratio= 0.4, random_seed=None):
    # Set seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Step 1: Get unique categories in each column
    unique_categories = {col: df[col].unique() for col in df.columns}

    # Step 2: Generate random distortion factors automatically for each category
    def generate_random_distortion_factors(unique_categories, low=low, high=high):
        """
        Generate random distortion factors for each category within a specified range.
        The distortion factors are used to create sampling weights for each category.
        This allows for taking non-representative samples.
        The distortion factors are uniformly distributed between `low` and `high`.
        Adjust `low` and `high` to control the distortion level.
        """
        distortion_factors = {}
        for col, categories in unique_categories.items():
            distortion_factors[col] = {category: np.random.uniform(low, high) for category in categories}
        return distortion_factors

    distortion_factors = generate_random_distortion_factors(unique_categories)

    # Step 3: Calculate sampling weights based on the randomly generated distortion factors
    sampling_weights = np.ones(len(df))  # Start with uniform weights

    for col, distortions in distortion_factors.items():
        for category, factor in distortions.items():
            # Apply distortion to weights for each category
            sampling_weights[df[col] == category] *= factor

    # Normalize weights to sum to 1
    sampling_weights /= sampling_weights.sum()

    # Step 4: Sample data using the distorted weights
    source_data = df.sample(frac=split_ratio, weights=sampling_weights, random_state=random_seed)

    # Step 5: Define target (population) data as the remaining data
    target_data = df.drop(source_data.index)

    return source_data, target_data


def align_categories(source_df, target_df):
    """
    Aligns categorical variables between two dataframes by keeping only categories
    present in both datasets. 
    This ensures that the categorical variables in both dataframes have the same
    categories, which is important for running the IPF procedure.
    
    Args:
        source_df (pd.DataFrame): First dataframe
        target_df (pd.DataFrame): Second dataframe
        
    Returns:
        tuple: (aligned_source_df, aligned_target_df)
    """
    # Create copies to avoid modifying original dataframes
    source = source_df.copy()
    target = target_df.copy()
    
    # Get categorical columns
    # cat_columns = source.select_dtypes(include=['category', 'object']).columns
    cat_columns = target.columns
    for col in cat_columns:
        if col in target.columns:
            # Get common categories
            source_cats = set(source[col].unique())
            target_cats = set(target[col].unique())
            common_cats = list(source_cats.intersection(target_cats))
            
            # Filter rows to keep only common categories
            source = source[source[col].isin(common_cats)]
            target = target[target[col].isin(common_cats)]
            
            # Convert to category type with aligned categories
            # source[col] = pd.Categorical(source[col], categories=common_cats)
            # target[col] = pd.Categorical(target[col], categories=common_cats)
    
    return source, target

# ------ IPF Data Input Functions and Sampling/Synthezising Functions ----------------

def ipf_target_in(target, source, columns, path, save_to_df=True):
    """
    Generate target margins for IPF from the target data.
    """ 
    target_ipf = target.copy()
    source_ipf = source.copy()
    for col in columns:
        unique, counts = np.unique(target_ipf[col], return_counts=True)
        # Remove values from target that are not in source
        unique_source = np.unique(source_ipf[col])
        unique_cleaned = list(unique)
        counts_cleaned = list(counts)
        for value in unique: # target
            if value not in unique_source:
                idx = unique_cleaned.index(value)
                unique_cleaned.pop(idx)
                counts_cleaned.pop(idx)
        df = pd.DataFrame(dict(zip(unique_cleaned, counts_cleaned)), index=[0])
        # Put 0 where data in source not in target
        for value in unique_source: # source
            if value not in unique:
                df[value] = 0
        df.to_csv(f"{path}/{col}.csv", index=False)


def sample_ipu(path, target_margin_dir, source_data_name="source", out_data_name="ipu-synthesized"):
    """
    Sample data using the Iterative Proportional Updating (IPU) method.
    IPU is a heuristic but fast implementation of the IPF algorithm.
    """
    # geo_level: the geographic level, either PUMA, County, or State
    # source_data_name: The name of the source data file
    # out_data_name: The name of the output data file

    # # Run R ipf script
    subprocess.call(
        ["Rscript", f"./ipu.R", path, target_margin_dir, source_data_name, out_data_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    sampled = pd.read_csv(f"{path}/out/{out_data_name}.csv")
    return sampled

def sample_seq_ipf(path, geo_level, source_data_name="source", out_data_name="seqipf-synthesized"):
    # # Run R ipf script
    subprocess.call(
        ["Rscript", f"{path}/seq_ipf.R", geo_level, source_data_name, out_data_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    # Generate samples from weights
    # sample = pd.read_csv(f"{path}/seq-ipf-synthesized.csv")
    sampled = pd.read_csv(f"data/out/{geo_level}/{out_data_name}.csv")

    return sampled


#----------- Evaluation metrics/statistics -------------
def compare_correlations(df, groups, corr_type="kendall"):
    """Compute within-group and between-group correlations
    Args:
        df: data frame (pd.DataFrame)
        groups: A list of group names or indices
    """
    corr_matrix = df.corr(corr_type)
    group_indices = groups
    # Print within-group correlations
    print("Average within-group correlations:")
    for i, group in enumerate(group_indices):
        group_corr = np.abs(corr_matrix.iloc[group, group]).mean().mean()
        print(f"Group {i+1}: {group_corr:.3f}")

    # Print between-group correlations
    print("\nAverage between-group correlations:")
    for i, g1 in enumerate(group_indices):
        for j, g2 in enumerate(group_indices[i+1:], i+1):
            corr = np.abs(corr_matrix.iloc[g1, g2]).mean().mean()
            print(f"Group {i+1} vs Group {j+1}: {corr:.3f}")


def cramers_v(x, y, bias_correction=True):
    """Calculate Cramer's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    # Apply bias correction to avoid overestimation
    if bias_correction:
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
    else:
        phi2corr = phi2
        rcorr = r
        kcorr = k
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def compute_cramers_v_matrix(df):
    """Compute a matrix of Cramer's V values for all pairs of columns in the dataframe."""
    cols = df.columns
    n = len(cols)
    cramers_v_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cramers_v_matrix[i, j] = cramers_v(df[cols[i]], df[cols[j]])
    return pd.DataFrame(cramers_v_matrix, index=cols, columns=cols)


def compute_avg_correlation(df, corr_type="spearman", digits=3):
    if corr_type == "cramers_v":
        corr_mat = compute_cramers_v_matrix(df)
    else:
        corr_mat = df.corr(method=corr_type)
    # np.tiu_indices_from(): Finds the indices of the upper triangle of the matrix, excluding the diagonal.
    return np.round(np.mean(np.abs(corr_mat.values[np.triu_indices_from(corr_mat, k=1)])), digits)

def macd(df1, df2, corr_type="spearman"):
    """Compute the mean absolute correlation difference (MACD) between two dataframes."""
    m = len(df1.columns)
    if corr_type == "cramersv":
        # For correlations in the range [0, 1]
        corr_mat = compute_cramers_v_matrix(df1)
        corr_mat2 = compute_cramers_v_matrix(df2)
        macd = np.sum(np.abs(corr_mat.values - corr_mat2.values))/(m*(m-1))
    else:
        # For correlations in the range [-1, 1]
        corr_mat = df1.corr(method=corr_type)
        corr_mat2 = df2.corr(method=corr_type)
        # Divide by 2 to avoid double counting, since the matrix is symmetric
        # This scales the MACD to be in the range [0, 1]
        macd = np.sum(np.abs(corr_mat.values - corr_mat2.values))/(2*m*(m-1))

    # unique_cors = corr_mat.values[np.triu_indices_from(corr_mat, k=1)] # extract upper triangle
    # unique_cors2 = corr_mat2.values[np.triu_indices_from(corr_mat2, k=1)] # extract upper triangle
    # macd = np.mean(np.abs(unique_cors - unique_cors2))
    return macd   


def compute_amae_akld(target_df, synthetic_df):
    """
    Computes the Average Mean Absolute Error (AMAE) and 
    Average Kullback-Leibler Divergence (AKLD) between the marginal 
    distributions of categorical variables in target and synthetic datasets.

    Parameters:
    - target_df (pd.DataFrame): Target dataset with categorical variables.
    - synthetic_df (pd.DataFrame): Synthetic dataset with categorical variables.

    Returns:
    - amae (float): Average Mean Absolute Error (AMAE).
    - akld (float): Average Kullback-Leibler Divergence (AKLD).
    """
    
    columns = target_df.columns  # Assume all columns are categorical
    m = len(columns)  # Number of categorical variables
    amae_list, akld_list = [], []

    for col in columns:
        # Compute marginal distributions for target and synthetic data
        target_dist = target_df[col].value_counts(normalize=True)
        synthetic_dist = synthetic_df[col].value_counts(normalize=True)

        # Ensure both distributions have the same categories
        common_categories = target_dist.index.union(synthetic_dist.index)
        target_probs = target_dist.reindex(common_categories, fill_value=0).values
        synthetic_probs = synthetic_dist.reindex(common_categories, fill_value=0).values

        c_j = len(common_categories)  # Number of categories for this variable
        # Compute Mean Absolute Error (MAE) for this variable
        mae_j = np.mean(np.abs(target_probs - synthetic_probs))
        amae_list.append(mae_j)

        # Compute KL Divergence for this variable (adding small epsilon to avoid log(0))
        epsilon = 1e-10
        kl_j = entropy(target_probs + epsilon, synthetic_probs + epsilon)  # KL divergence
        akld_list.append(kl_j)

    # Compute AMAE and AKLD by averaging over all variables
    amae = np.sum(amae_list)
    akld = np.sum(akld_list)

    return {'AMAE': amae, 'AKLD': akld} # amae, akld

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
    Args:
        data1: reference data
        data2: synthetic data
        columns: columns to project over
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
        srmse_dict["SRMSE "+str(i)] = SRMSE
    return srmse_dict

# Codes for the paper “Scalable Population Synthesis with Deep Generative Modeling”: https://github.com/stasmix/popsynth


