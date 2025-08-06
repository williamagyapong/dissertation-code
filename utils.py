import numpy as np
import pandas as pd
import itertools
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import norm
import scipy.stats as stats
from sklearn.preprocessing import OneHotEncoder
import dcor # Distance correlation

from copula_tools import CopulaScaler
# from pomegranate.bayesian_network import BayesianNetwork # (latest version API)

# from pomegranate import BayesianNetwork 
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata # Deprecated!
from sdv.metadata import Metadata
from sdmetrics.single_table import NewRowSynthesis


def set_seed(seed=123):
    # Set a random seed for NumPy
    np.random.seed(123)
    # Set a random seed for Python's built-in random module
    import random
    random.seed(123)
    return None 


from pandas.api.types import CategoricalDtype

def convert_columns_to_categorical(df, columns=None, order_info=None):
    """
    Convert specified columns (or all if None) in df to categorical dtype,
    inferring categories from the data and applying ordering from order_info. 
    If all columns are used and order info is not present for a given column, 
    it will be marked as unordered. 

    Parameters:
    - df: pandas DataFrame
    - columns: list of column names to convert (default: all columns)
    - order_info: dict with column names as keys and either:
        - True/False for ordering
        - Or a dict with keys 'ordered' and optional 'categories' (to override)

    Returns:
    - df copy with converted categorical columns
    """
    df = df.copy()

    if columns is None:
        columns = df.columns

    for col in columns:
        if col not in df.columns:
            continue

        col_data = df[col]
        info = order_info.get(col, {}) if order_info else {}

        # If user passed just True/False for ordering
        if isinstance(info, bool):
            ordered = info
            categories = sorted(col_data.dropna().unique()) if ordered else col_data.dropna().unique()
        # If user passed a dict
        elif isinstance(info, dict):
            ordered = info.get('ordered', False)
            if 'categories' in info:
                categories = info['categories']
            else:
                categories = sorted(col_data.dropna().unique()) if ordered else col_data.dropna().unique()
        else:
            ordered = False
            categories = col_data.dropna().unique()

        cat_type = CategoricalDtype(categories=categories, ordered=ordered)
        df[col] = df[col].astype(cat_type)

    return df


def get_variable_names_by_group(variable_names, group_sizes):
    """
    Generate a list of lists containing variable names for each group.
    
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
    Simulate ordinal categorical data such that variables within a group are correlated and sampled 
    from a multivariate normal distribution and binned into categories.

    Args:
        num_samples: Number of samples (rows) in the dataset.
        num_groups: Number of independent groups.
        group_sizes: A list with the number of variables in each group.
        categories_per_group: A list specifying the number of categories for each group.
                              If None, defaults to 2 categories per group.
        correlation_strength: Correlation strength for within-group dependency (default=0.8).

    Returns:
        A pandas DataFrame with simulated dependent categorical data.
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



def source_target_split(df, low=0.5, high=1.5, split_ratio= 0.4, random_seed=None):
    # Set seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Step 1: Get unique categories in each column
    unique_categories = {col: df[col].unique() for col in df.columns}

    # Step 2: Generate random distortion factors automatically for each category
    def generate_random_distortion_factors(unique_categories, low=low, high=high):
        """
        Generate random distortion factors for each category within a specified range.
        Adjust `low` and `high` to control the distortion level.
        """
        distortion_factors = {}
        for col, categories in unique_categories.items():
            distortion_factors[col] = {category: np.random.uniform(low, high) for category in categories}
        return distortion_factors

    # Automatically generated distortion factors
    distortion_factors = generate_random_distortion_factors(unique_categories)

    # Step 3: Calculate sampling weights based on the randomly generated distortion factors
    sampling_weights = np.ones(len(df))  # Start with uniform weights

    for col, distortions in distortion_factors.items():
        for category, factor in distortions.items():
            # Apply distortion to weights for each category
            sampling_weights[df[col] == category] *= factor

    # Normalize weights to sum to 1
    sampling_weights /= sampling_weights.sum()

    # Step 4: Sample source (sample) data using the distorted weights
    source_data = df.sample(frac=split_ratio, weights=sampling_weights, random_state=random_seed)

    # Step 5: Define target (population) data as the remaining data
    target_data = df.drop(source_data.index)

    return source_data, target_data


def align_categories(source_df, target_df):
    """
    Aligns categorical variables between two dataframes by keeping only categories
    present in both datasets.
    
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
    # Ensure all columns are categorical
    df = df.copy()
    df = df.astype('category')
    n = len(cols)
    cramers_v_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cramers_v_matrix[i, j] = cramers_v(df[cols[i]], df[cols[j]])
    return pd.DataFrame(cramers_v_matrix, index=cols, columns=cols)

# Compute Mutual Information matrix

from sklearn.metrics import mutual_info_score

def normalized_mutual_info(x, y):
    mi = mutual_info_score(x, y)
    hx = mutual_info_score(x, x)  # Entropy of x
    hy = mutual_info_score(y, y)  # Entropy of y
    denom = np.sqrt(hx * hy)
    return mi / denom if denom > 0 else 0


def compute_mi_matrix(df):
    """
    Compute pairwise mutual information between all categorical columns in a DataFrame.
    Returns a symmetric matrix.
    """
    cols = df.columns
    n = len(cols)
    mi_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            x = df[cols[i]]
            y = df[cols[j]]
            # mi = mutual_info_score(x, y)
            mi = normalized_mutual_info(x, y)
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi  # symmetric

    return pd.DataFrame(mi_matrix, index=cols, columns=cols)


def ddcor(x, y):
    """Compute distance correlation between two discrete (categorical variables)"""
    pass 
    # x_mat = OneHotEncoder().fit_transform(df_sample[['age']]).toarray()

    # y_mat = OneHotEncoder().fit_transform(df_sample[['income']]).toarray()

def dcor_matrix(df):
    """compute a correlation matrix across all variables in df based on distance correlation"""
    n = df.shape[1]
    cols = df.columns.tolist()
    ddcor_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_encoded = OneHotEncoder().fit_transform(df[[cols[i]]]).toarray()
            y_encoded = OneHotEncoder().fit_transform(df[[cols[j]]]).toarray()
            ddcor_matrix[i, j] = dcor.distance_correlation(x_encoded, y_encoded)
    return pd.DataFrame(ddcor_matrix, index=cols, columns=cols)

def plot_corr_matrix(df, corr_type="spearman", title="", vmin=-1, vmax=1, figsize=(8, 6), save_to_file=None):
    """Plot a correlation matrix as a heatmap."""
    # Compute correlation matrix 
    if corr_type == "cramers_v":
        corr_matrix = compute_cramers_v_matrix(df)
    elif corr_type == "mi":
        corr_matrix = compute_mi_matrix(df)
    else:
        corr_matrix = df.corr(method=corr_type)
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix,
                 annot=True, 
                 cmap='coolwarm', 
                 fmt=".2f",
                 linewidths=0.5,
                 annot_kws={"size": 6},
                 vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    
    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight') # , dpi=300, not really needed for pdf formats
    plt.show()

def compute_avg_correlation(df, corr_type="spearman", digits=3):
    if corr_type == "cramers_v":
        corr_mat = compute_cramers_v_matrix(df)
    else:
        corr_mat = df.corr(method=corr_type)
    # np.tiu_indices_from(): Finds the indices of the upper triangle of the matrix, excluding the diagonal.
    return np.round(np.mean(np.abs(corr_mat.values[np.triu_indices_from(corr_mat, k=1)])), digits)

def macd(df1, df2, corr_type="spearman", digits=3):
    """Compute the mean absolute correlation difference (MACD) between two dataframes."""
    m = len(df1.columns)
    if corr_type == "cramersv":

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
    return np.round(macd, digits)    


def ipf_in(target, source, columns, path):
    target_ipf = target.copy()
    source_ipf = source.copy()
    if "WIF" in target_ipf:  
        target_ipf["WIF"] += 1
        source_ipf["WIF"] += 1
    source_ipf.to_csv(f"{path}/source.csv", index=False)
    for col in columns:
        unique, counts = np.unique(target_ipf[col], return_counts=True)
        # Remove values from target that are not in source
        unique_source = np.unique(source_ipf[col])
        unique_cleaned = list(unique)
        counts_cleaned = list(counts)
        for value in unique:
            # These could be sampling zeros, can we deal with this without droping them?
            if value not in unique_source:
                idx = unique_cleaned.index(value)
                unique_cleaned.pop(idx)
                counts_cleaned.pop(idx)
        df = pd.DataFrame(dict(zip(unique_cleaned, counts_cleaned)), index=[0])
        # Put 0 where data in source not in target
        for value in unique_source:
            if value not in unique:
                df[value] = 0
        df.to_csv(f"{path}/{col}.csv", index=False)

def ipf_target_in(target, source, columns, path, save_to_df=True):
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
        # print(df)
        # print(counts_cleaned)
        # margins.append(np.array(counts_cleaned))
        # margins[col] = counts_cleaned
        for value in unique_source:
            if value not in unique:
                df[value] = 0
        # df.to_csv(f"{path}/{col}.csv", index=False)
        # create this margins dict for use with pyipu
        df.columns = df.columns.astype(str)
        margins[col] = df
    return margins

def sample_ipf(path, target_margin_dir, source_data_name="source", out_data_name="ipu-synthesized"):
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


def sample_copula(source, target, sampler, sampler_args={}):
    source_scaler = CopulaScaler()
    target_scaler = CopulaScaler()
    source_scaler.fit(source)
    target_scaler.fit(target)
    source_tr = source_scaler.transform(source)
    # sample = sampler(source_tr, len(target), **sampler_args)
    sample, _ = sampler(source_tr, len(target), **sampler_args)
    sample = source_scaler.interpolation(sample, source.columns)
    return target_scaler.inverse_transform(sample)

#------ Use copula directly to any data source

class CategoricalEncoder:
    """Assign numeric labels and be able to reverse the process 
    """
    def __init__(self):
        self.encoders = {}  # To store mapping dictionaries

    def fit_transform(self, df):
        df_encoded = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
                unique_vals = df[col].astype(str).unique()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                reverse_mapping = {idx: val for val, idx in mapping.items()}
                self.encoders[col] = {'forward': mapping, 'backward': reverse_mapping}
                df_encoded[col] = df[col].map(mapping)
        return df_encoded

    def inverse_transform(self, df_encoded):
        df_decoded = df_encoded.copy()
        for col, mapping in self.encoders.items():
            df_decoded[col] = df_encoded[col].map(mapping['backward'])
        return df_decoded
    
def apply_copula(source, target):
    # Encode categorical features
    source_encoder = CategoricalEncoder()
    target_encoder = CategoricalEncoder()
    source_encoded = source_encoder.fit_transform(source)
    target_encoded = target_encoder.fit_transform(target)

    source_scaler = CopulaScaler()
    target_scaler = CopulaScaler()
    source_scaler.fit(source_encoded)
    target_scaler.fit(target_encoded)
    source_tr = source_scaler.transform(source_encoded)
    sample = source_scaler.interpolation(source_tr, source.columns)
    sample = target_scaler.inverse_transform(sample)
    # use the source encoder to map numeric labels back to the original labels
    sample = target_encoder.inverse_transform(sample)

    return sample


# def sample_bn(source, n, n_jobs=1):
#     bn = BayesianNetwork.from_samples(source, algorithm="greedy", n_jobs=n_jobs)
#     sample = pd.DataFrame(bn.sample(n, algorithm="rejection"), columns=source.columns)
#     return sample

# def sample_bn(source, n, n_jobs=1):
#     bn = BayesianNetwork()
#     # bn = BayesianNetwork.from_samples(source, algorithm="greedy", n_jobs=n_jobs)
#     BayesianNetwork.fit(source, algorithm="greedy", n_jobs=n_jobs)
#     sample = pd.DataFrame(bn.sample(n, algorithm="rejection"), columns=source.columns)
#     return sample

# Learn a Bayesian network from the synthetic data
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import HillClimbSearch
# from pgmpy.estimators import GES 
from pgmpy.estimators import TreeSearch

# This implementation tends to overfit the data??
# TODO: Use a more robust method to learn the structure of the Bayesian network
def learn_bn_from_data(data, weighted=False, **hc_args):
    """
    Learn a Bayesian Network from the given data using Hill Climb Search.  
    """
#  hc_args = {"scoring_method": "bic-d", "max_indegree": 2}):
    # scoring_method="bic-d", max_indegree=2
    est = HillClimbSearch(data) # Best with k2 scoring method
    # est = GES(data) # GES not significantly different from HCS
    # est = TreeSearch(data)
    best_model = est.estimate(scoring_method="bic-d", **hc_args, show_progress=False) # 
    model = BayesianNetwork(best_model.edges())
    model.fit(data, estimator=MaximumLikelihoodEstimator, weighted=weighted)
    return model

# Generate synthetic data from the learned Bayesian network
from pgmpy.sampling import BayesianModelSampling


def sample_from_bn(data, n_samples, weighted=False, **hc_args):
    """
    Sample synthetic data from a Bayesian Network model.
    
    Parameters:
    - data: the real data to learn from.
    - n_samples: Number of samples to generate.
    
    Returns:
    - DataFrame containing the synthetic samples.
    """

    # Learn the Bayesian network from the data
    model = learn_bn_from_data(data, weighted=weighted, **hc_args)
    sampler = BayesianModelSampling(model)
    return sampler.forward_sample(size=n_samples), model




def sample_ctgan(source, n, ctgan_args={}):
    # metadata = SingleTableMetadata()
    metadata = Metadata.detect_from_dataframe(data=source)
    ctgan = CTGANSynthesizer(metadata, **ctgan_args)
    ctgan.fit(source)
    return ctgan.sample(num_rows=n)


def sample_tvae(source, n, tvae_args={}):
    # metadata = SingleTableMetadata()
    metadata = Metadata.detect_from_dataframe(data=source)
    tvae = TVAESynthesizer(metadata, **tvae_args)
    tvae.fit(source)
    return tvae.sample(num_rows=n)

def sample_gaussiancopula(source, n, gc_args={}):
    metadata = Metadata.detect_from_dataframe(data=source)
    gc = GaussianCopulaSynthesizer(metadata, **gc_args)
    gc.fit(source)
    return gc.sample(num_rows=n)


def sample_independent(source, target):
    columns = source.columns
    # Independent baseline
    ind_data = np.zeros(shape=target.shape)
    for i in range(ind_data.shape[0]):
        for j in range(ind_data.shape[1]):
            ind_data[i,j] = source[columns[j]].sample(1)
    return pd.DataFrame(ind_data, columns=columns)

from scipy.stats import entropy

def compute_amae_akld(target_df, synthetic_df):
    """
    Computes the Average Mean Absolute Error (AMAE) and 
    Average Kullback-Leibler Divergence (AKLD) between the marginal 
    distributions of categorical variables in target and synthetic datasets.

    Parameters:
    - target_df (pd.DataFrame): Target dataset with categorical variables.
    - synthetic_df (pd.DataFrame): Synthetic dataset with categorical variables.

    Returns:
    - am (float): Average Mean Absolute Error (AMAE).
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

def srmse(data1, data2):
    """ Compute Standardized Root Mean Squared Error between two datasets.

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

def diversity_score(real_df, synthetic_df, metadata=None, 
                    sample_size=10000):
    if metadata is None: # Automatically detect the column types
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=real_df)
    # if metadata is None:
    #     metadata = Metadata.detect_from_dataframe(data=real_df)
    # print(metadata)

    # Determine the sample size to speed up the computation for large datasets
    sample_size = min(sample_size, len(synthetic_df))
    print(f"Sample size for diversity score computation: {sample_size}")
    score = NewRowSynthesis.compute(
        real_data = real_df,
        synthetic_data = synthetic_df,
        metadata = metadata,
        synthetic_sample_size = sample_size
    )
    return score
    

def result_table(source, target, synthetic_data, columns, max_projection=5, save=False, diversity=False, zeros=False, macd_score=False, syn_size=10000, metadata=None, digits=4, path=""):
    # Detect data types for each column
    # metadata = Metadata.detect_from_dataframe(data=source)
    # metadata.detect_from_dataframe(data=source)
    
    # Calculate SRMSE and zeros/diversity
    result_dict = {}
    for model in synthetic_data:
        result_dict[model] = {}
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
            result_dict[model]["SRMSE "+str(i)] = SRMSE
        # srmse_dict[model]["Zeros"] = sampling_zeros(source, target, df)

        if macd_score:
            # Compute MACD for each correlation type
            # result_dict[model]["MACD_S"] = macd(source, df, corr_type="spearman")
            # result_dict[model]["MACD_K"] = macd(source, df, corr_type="kendall")
            result_dict[model]["MACD_C"] = macd(source, df, corr_type="cramersv")

        if diversity:
            # print(metadata)
            result_dict[model]["Diversity"] = diversity_score(source, df, sample_size=syn_size, metadata=metadata)

        if zeros:
            result_dict[model]["Zeros"] = sampling_zeros(source, target, df)

    table = []
    for model in result_dict:
        table.append(
            pd.DataFrame({i:result_dict[model][i] for i in result_dict[model]}, index=[model]))
    table = pd.concat(table).round(digits)
    
    if save:
        # "path" must include name of the file without the extension
        table.to_csv(f"{path}.csv")
        
    return table


# --- Synthetic Population Generation Experiments
import ipf_utils
import timeit

def run_syn_pop_experiments(source, target, feature_groups, baseline=True, run_ipf=True, run_ipu=True, path=None, n_srmse=1):
    set_seed(123)
    selected_features = list(target.columns)

    runtimes = []
    # sample an independent baseline
    if baseline:
        print("Fitting the baseline....\n")
        start_time = timeit.default_timer()
        ind_syn =  sample_independent(source, target) 
        end_time = timeit.default_timer()
        runtimes.append(end_time - start_time)

    # Apply the standard IPF
    if run_ipf:
        print("Running IPF ....\n")
        start_time = timeit.default_timer()
        ipf_syn, _, _ = ipf_utils.std_ipf(source, target, selected_features) 
        end_time = timeit.default_timer()
        ipf_runtime = end_time - start_time
        runtimes.append(ipf_runtime)
    
    # Apply the IPU implementation of IPF: Agyapong (2025): pyipu
    if run_ipu:
        print("Running IPU ....\n")
        start_time = timeit.default_timer()
        target_margins = ipf_utils.ipu_target_in(target, source, selected_features)
        ipu_syn, _ = ipf_utils.ipu_syn(source, target_margins)
        end_time = timeit.default_timer()
        ipu_runtime = end_time - start_time
        runtimes.append(ipu_runtime)
    
    # run the proposed BIPF (Blockwise IPF): Dissertation (Agyapong 2025)
    #--- SBIPF
    print("Running SBIPF ....\n")
    start_time = timeit.default_timer()
    target_margins_dict = ipf_utils.ipf_target_in(target, source, selected_features)
    sbipf_syn_gb = ipf_utils.sbipf(source, target_margins_dict, feature_groups, ipf_utils.ipfn_wrapper, n=target.shape[0])
    end_time = timeit.default_timer()
    sbipf_gb_runtime = end_time - start_time
    runtimes.append(sbipf_gb_runtime)

    # --- NBIPF
    print("Running NBIPF ....\n")
    start_time = timeit.default_timer()
    target_margins_dict = ipf_utils.ipf_target_in(target, source, selected_features)
    nbipf_syn_gb, _ = ipf_utils.nbipf(source, target_margins_dict, feature_groups, ipf_utils.ipfn_wrapper, n=target.shape[0])
    end_time = timeit.default_timer()
    nbipf_gb_runtime = end_time - start_time
    runtimes.append(nbipf_gb_runtime)

    # Use random partitioning for SBIPF
    print("Running SBIPF with random partition ....\n")
    start_time = timeit.default_timer()
    target_margins_dict = ipf_utils.ipf_target_in(target, source, selected_features)
    naive_feature_seq = ipf_utils.group_features(selected_features, n_group=len(feature_groups), random_seed=123)
    sbipf_syn_rp = ipf_utils.sbipf(source, target_margins_dict, naive_feature_seq, ipf_utils.ipfn_wrapper, n=target.shape[0])
    end_time = timeit.default_timer()
    sbipf_rp_runtime = end_time - start_time
    runtimes.append(sbipf_rp_runtime)

    print("Running NBIPF with random partition ....\n")
    start_time = timeit.default_timer()
    target_margins_dict = ipf_utils.ipf_target_in(target, source, selected_features)
    naive_feature_seq = ipf_utils.group_features(selected_features, n_group=len(feature_groups), random_seed=123)
    nbipf_syn_rp, _ = ipf_utils.nbipf(source, target_margins_dict, naive_feature_seq, ipf_utils.ipfn_wrapper, n=target.shape[0])
    end_time = timeit.default_timer()
    nbipf_rp_runtime = end_time - start_time
    runtimes.append(nbipf_rp_runtime)

    # Run Bayesian Network
    print("Running BN ....\n")
    start_time = timeit.default_timer()
    score_method = 'bic-d' # k2 < aic-d < bic-d (k2 helps with higher values of SRMSE, but very poor diversity scores)
    # Stick to 'bic-d' with HillClimbSearch
    bn_syn, _ = sample_from_bn(source, n_samples= target.shape[0]) 
    end_time = timeit.default_timer()
    runtimes.append(end_time - start_time)
    

    # Run BN Copula (Jutras-Dube et al, 2024): Copula-based transferable models for syn pop generation
    print("Running BN with Copula normalization ....\n")
    start_time = timeit.default_timer()
    bn_copula_syn = sample_copula(source, target, sample_from_bn)
    end_time = timeit.default_timer()
    runtimes.append(end_time - start_time)

    # Apply Hybrid method (IPF + BN): Dissertation (Agyapong 2025)
    if run_ipf:
        print("Running IPF + BN ....\n")
        start_time = timeit.default_timer()
        ipf_bn_syn, _ = sample_from_bn(ipf_syn, n_samples= target.shape[0])
        end_time = timeit.default_timer()
        runtimes.append(ipf_runtime + (end_time - start_time))

    if run_ipu:
        print("Running IPU + BN ....\n")
        start_time = timeit.default_timer()
        ipu_bn_syn, _ = sample_from_bn(ipu_syn, n_samples= target.shape[0])
        end_time = timeit.default_timer()
        runtimes.append(ipu_runtime + (end_time - start_time))

    print("Running SBIPF + BN ....\n")
    start_time = timeit.default_timer()
    sbipf_bn_syn_gb, _ = sample_from_bn(sbipf_syn_gb, n_samples= target.shape[0])
    end_time = timeit.default_timer()
    runtimes.append(sbipf_gb_runtime + (end_time - start_time))

    print("Running NBIPF + BN ....\n")
    start_time = timeit.default_timer()
    nbipf_bn_syn_gb, _ = sample_from_bn(nbipf_syn_gb, n_samples= target.shape[0])
    end_time = timeit.default_timer()
    runtimes.append(nbipf_gb_runtime + (end_time - start_time))

    print("Running SBIPF-RP + BN ....\n")
    start_time = timeit.default_timer()
    sbipf_bn_syn_rp, _ = sample_from_bn(sbipf_syn_rp, n_samples= target.shape[0])
    end_time = timeit.default_timer()
    runtimes.append(sbipf_rp_runtime + (end_time - start_time))

    print("Running NBIPF-RP + BN ....\n")
    start_time = timeit.default_timer()
    nbipf_bn_syn_rp, _ = sample_from_bn(nbipf_syn_rp, n_samples= target.shape[0])
    end_time = timeit.default_timer()
    runtimes.append(nbipf_rp_runtime + (end_time - start_time))


    # Generate results
    print('Now generating results...\n')
    synthetic_df = {
        **({"Independent": ind_syn} if baseline else {}),
        **({"IPF": ipf_syn} if run_ipf else {}),
        **({"IPU": ipu_syn} if run_ipu else {}),
        "SBIPF-GB": sbipf_syn_gb,
        "NBIPF-GB": nbipf_syn_gb,
        "SBIPF-RP": sbipf_syn_rp,
        "NBIPF-RP": nbipf_syn_rp,
        "BN": bn_syn,
        "BN Copula": bn_copula_syn,
        **({"IPF BN": ipf_bn_syn} if run_ipf else {}),
        **({"IPU BN": ipu_bn_syn} if run_ipu else {}),
        "SBIPF-GB BN": sbipf_bn_syn_gb,
        "NBIPF-GB BN": nbipf_bn_syn_gb,
        "SBIPF-RP BN": sbipf_bn_syn_rp,
        "NBIPF-RP BN": nbipf_bn_syn_rp
    }

    # Detect meta data from the source data
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=source)
    result = result_table(source, target, synthetic_df, set(selected_features), metadata=metadata,
                                 max_projection=n_srmse, macd_score=True, diversity=True
        )
    
    # Concatenate the runtimes 
    runtimes_df = pd.DataFrame(runtimes, columns=['Runtime'], index= result.index).round(4)
    result = pd.concat([result, runtimes_df], axis=1)
    # Save results
    if path is not None:
        result.to_csv(path)
        print(f"Results saved to '{path}'")
    return result

#--- Visualizations

def margin_bar_plot(data, color_pal):

    # Prepare data for plotting
    data_combined = pd.DataFrame()
    temp_df = []
    variables = list(data['Target'].columns)
    for df_name in data:
        df = data[df_name]
        for variable in variables:
            counts = df[variable].value_counts(normalize=True).reset_index()
            counts.columns = ['value', 'proportion']
            counts['dataset'] = df_name
            counts['variable'] = variable
            data_combined = pd.concat([data_combined, counts])

    # Calculate the number of rows and columns for the grid
    ncols = 2  # Fixed number of columns
    variable_count = len(variables)
    nrows = math.ceil(variable_count / ncols)  # Determine rows based on variable count

    # Set the figure size dynamically
    my_figsize = (ncols * 9, nrows * 4.8)  # Adjust dimensions for better scaling
    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=my_figsize) #
    axes = axes.flatten() 

    # # Add bar plots to the subplots
    for i, variable in enumerate(variables):
        subset = data_combined[data_combined['variable'] == variable]
        sns.barplot(ax=axes[i], data=subset, x='value', y='proportion', hue='dataset', palette=color_pal)
        axes[i].set_title(variable)
        axes[i].set_ylabel('')
        axes[i].set_xlabel('')
        axes[i].legend().set_visible(False)
    axes[1].legend(loc='upper right')
    # Adjust layout
    plt.tight_layout()
    # plt.suptitle('Comparative Bar Chart and Histogram of Marginal Counts for Each Variable', y=1.02)
    # plt.savefig("../data/IPEDS/out/margins.png")
    plt.show()
    return None

#------ Machine Learning Efficacy Metrics ---------
## Feature importance
# Function to extract original variable name from encoded feature name
import re
def extract_original_feature(encoded_feature_name):
    # This pattern assumes the format is "original_feature_name_category_value"
    # Adjust the regex if feature names follow a different pattern
    match = re.match(r'([^_]+)_.*', encoded_feature_name)
    if match:
        return match.group(1)
    return encoded_feature_name

def get_feature_importance(cat_encoder, model):
    encoded_feature_names = cat_encoder.named_steps['onehot'].get_feature_names_out(list(cat_encoder.feature_names_in_))
    feature_importance_df = pd.DataFrame({
        'feature': encoded_feature_names,
        'importance': model.feature_importances_
    })

    feature_importance_df['original_feature'] = feature_importance_df['feature'].apply(extract_original_feature)

    # Aggregate importances by original feature
    aggregated_importance_df = feature_importance_df.groupby('original_feature')['importance'].sum().reset_index()
    aggregated_importance_df = aggregated_importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    return aggregated_importance_df
