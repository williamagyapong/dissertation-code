"""
This module provides utility functions for conducting simulation studies in parallel
as part of a PhD dissertation research. It includes tools for setting up, running, 
and analyzing simulations, as well as any necessary helper functions to streamline 
the research process.

The utilities are designed to be flexible and reusable, enabling efficient experimentation 
and data analysis for the dissertation's simulation-based studies.
"""

#----- Imports
# Import required libraries
import numpy as np
import pandas as pd
import timeit
import psutil
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import functools
import itertools
from tqdm import tqdm
# Add parent directory to the path
import sys
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(parent_dir)
# print(os.getcwd()) # Check the current working directory
import utils
import ipf_utils

#----- Simulation functions
def package_sim_results(sim, group_sizes, categories_per_group, rho, method, max_project,
                         reference_df, target_df, syn_df, start_time, stop_time):
    
    selected_features = target_df.columns.tolist()
    return {   
        "sim_run": sim + 1,
        "groups": len(group_sizes),
        "categories_per_group": categories_per_group[0], # works for uniform categories per group
        "dependency_level": rho,
        "method": method,
        **utils.project_srmse(target_df, syn_df, set(selected_features), max_project),
        **utils.compute_amae_akld(target_df, syn_df),
        "MACD_S": utils.macd(reference_df, syn_df, corr_type='spearman'),
        "MACD_C": utils.macd(reference_df, syn_df, corr_type='cramersv'),
        "runtime": stop_time - start_time
    }

def run_single_parameter_combination(params, ref_ratio=0.2,
                                    num_samples=10000, mode=None, tol=1e-6, run_ipf=True,
                                    max_project=1, nruns=1):
    """
    Run a single parameter combination (sim, categories_per_group, rho).
    
    This function is designed to be run in parallel for each parameter combination.
    """
    sim, group_sizes, categories_per_group, rho = params
    # Ensure categories_per_group is a list of the same length as group_sizes
    # Assigns uniform categories to all variables in the group and across groups
    categories_per_group = [categories_per_group]*len(group_sizes) 
    
    # Set a different random seed for each process to ensure independence
    np.random.seed(123 + sim * 1000 + hash(str(group_sizes)) % 1000 + hash(str(categories_per_group)) % 1000 + int(rho * 100))
    
    # Generate data
    if mode is None:
        df = utils.simulate_independent_groups(
            num_samples=num_samples, 
            group_sizes=group_sizes, 
            categories_per_group=categories_per_group, 
            rho=rho
        )
    elif mode == 'mvn':
        df, _ = utils.simulate_independent_groups_mvn(
            num_samples=num_samples, 
            group_sizes=group_sizes, 
            categories_per_group=categories_per_group, 
            rho=rho
        )
    elif mode == 'mvn2':
        df, _ = utils.simulate_relaxed_independent_groups_mvn(
            num_samples=num_samples,
            group_sizes=group_sizes,
            categories_per_group=categories_per_group,
            within_group_rho=rho,
        )
        
    # Split data into ref and target
    reference_df, target_df = utils.ref_target_split(df, split_ratio=ref_ratio)
    selected_features = target_df.columns.tolist()
    
    # Results for this parameter combination
    results = []
    
    # Store baseline results for the reference data
    results.append(
        package_sim_results(sim, group_sizes, categories_per_group, rho, "Reference", max_project, reference_df, target_df, reference_df, np.nan, np.nan)
    )

    # Apply algorithms
    if run_ipf:
        start_time = timeit.default_timer()
        std_ipf_syn, _ = ipf_utils.std_ipf(ref_set=reference_df, target_set=target_df, conv_rate=tol,
                                selected_features=selected_features) 
        stop_time = timeit.default_timer()
        results.append(
            package_sim_results(sim, group_sizes, categories_per_group, rho, "IPF", max_project, reference_df, target_df, std_ipf_syn, start_time, stop_time)
        )
    else:
        results.append(
            {   
                "sim_run": sim + 1,
                "groups": len(group_sizes),
                "categories_per_group": categories_per_group[0],
                "dependency_level": rho,
                "method": "IPF",
                **{'SRMSE 1': np.nan},
                **{'AMAE': np.nan, 'AKLD': np.nan},
                "MACD_S": np.nan,
                "MACD_C": np.nan,
                "runtime": np.nan
            }
        )

    # Apply blockwise IPF on true feature groups
    target_margins_dict = ipf_utils.ipf_target_in(target_df, reference_df, target_df.columns)
    
    # N-BIPF
    feature_seq = utils.get_variable_names_by_group(selected_features, group_sizes) # determine the feature sequence
    start_time = timeit.default_timer()
    nbipf_syn, _ = ipf_utils.nbipf(reference_df, target_margins_dict, feature_seq, ipf_utils.ipfn_wrapper, ipf_args={'convergence_rate':tol}, n=target_df.shape[0], runs=1)
    stop_time = timeit.default_timer()
    results.append(
        package_sim_results(sim, group_sizes, categories_per_group, rho, "N-BIPF", max_project, reference_df, target_df, nbipf_syn, start_time, stop_time)
    )

    # S-BIPF
    feature_seq = utils.get_variable_names_by_group(selected_features, group_sizes)
    start_time = timeit.default_timer()
    sbipf_syn = ipf_utils.sbipf(reference_df, target_margins_dict, feature_seq, ipf_utils.ipfn_wrapper, ipf_args={'convergence_rate':tol}, n=target_df.shape[0], runs=nruns)
    stop_time = timeit.default_timer()
    results.append(
        package_sim_results(sim, group_sizes, categories_per_group, rho, "S-BIPF", max_project, reference_df, target_df, sbipf_syn, start_time, stop_time)
    )

    # N-BIPF-RP
    start_time = timeit.default_timer()
    nbipf_rp_syn, _ = ipf_utils.nbipf(reference_df, target_margins_dict, feature_seq, ipf_utils.ipfn_wrapper, ipf_args={'convergence_rate':tol}, n=target_df.shape[0], runs=1)
    stop_time = timeit.default_timer()
    results.append(
        package_sim_results(sim, group_sizes, categories_per_group, rho, "N-BIPF-RP",
                             max_project, reference_df, target_df, nbipf_rp_syn, start_time, stop_time)
    )

    # S-BIPF-RP
    feature_seq = ipf_utils.group_features(selected_features, n_group=len(group_sizes), random_seed=123)
    start_time = timeit.default_timer()
    sbipf_rp_syn = ipf_utils.sbipf(reference_df, target_margins_dict, feature_seq, ipf_utils.ipfn_wrapper, ipf_args={'convergence_rate':tol}, n=target_df.shape[0], runs=nruns)
    stop_time = timeit.default_timer()
    results.append(
        package_sim_results(sim, group_sizes, categories_per_group, rho, "S-BIPF-RP",
                             max_project, reference_df, target_df, sbipf_rp_syn, start_time, stop_time)
    )
    
    return results


def run_sim(scenarios, ref_ratio=0.2,
             num_samples=10000, mode=None, tol=1e-6, run_ipf=True,
               nsims=1, max_project=1, nruns=1, free_cores=3):
    """ 
    Args:
    scenarios: Dictionary containing the dependency levels, group_sizes, and categories per group
    group_sizes: List of lists of integers representing the sizes of each group for different scenarios for a given set of variables p (e.g., [[2, 2, 2, 2, 2], [5, 5]] for p=10)
    dependency_levels: List of floats representing the dependency levels for different scenarios
    categories_per_group: List of lists of integers representing the number of categories for each group for different scenarios
    ref_ratio: Ratio of reference data to total data
    num_samples: Total number of samples to generate
    max_project: Maximum number of iterations for the projection
    tol: Convergence tolerance for the IPF algorithm
    run_ipf: Boolean indicating whether to run the IPF algorithm
    nruns: Number of runs for the IPF algorithm
    mode: Type of data distribution for the latent variables, multivariate normal (mvn), ..., etc
    nsims: Number of simulation runs to average results over

    Returns:
    A dictionary containing:
    - detailed_results: DataFrame with detailed results for each simulation run
    - avg_results: DataFrame with average results across all simulation runs
    - std_results: DataFrame with standard deviation of results across all simulation runs
    - final_results: DataFrame with average results and standard deviations
    """
    # Generate all parameter combinations
    all_combinations = list(itertools.product(
        range(nsims),
        scenarios['group_sizes'],
        scenarios['categories_per_group'],
        scenarios['dependency_levels']
    ))
    
    # Determine the number of CPU cores to use
    num_cores = mp.cpu_count()
    num_cores_to_use = max(1, num_cores - free_cores)  # Leave some cores free for the OS
    print(f"Running {len(all_combinations)} parameter combinations using {num_cores_to_use}/{num_cores} CPU cores")
    
    # Create a partial function with all parameters except the combination
    run_param_partial = functools.partial(
        run_single_parameter_combination,
        ref_ratio=ref_ratio,
        num_samples=num_samples,
        mode=mode,
        tol=tol,
        run_ipf=run_ipf,
        max_project=max_project,
        nruns=nruns
    )
    
    # Run parameter combinations in parallel
    all_results = []
    with ProcessPoolExecutor(max_workers=num_cores_to_use) as executor:
        # Map the function to all parameter combinations
        for result in tqdm(executor.map(run_param_partial, all_combinations), total=len(all_combinations), desc="Processing parameter combinations"):
            all_results.extend(result)
    
    # Convert all results to a DataFrame
    combined_results = pd.DataFrame(all_results)
    combined_results.rename(columns={
        'SRMSE 1': 'SRMSE'}, inplace=True)
    # Calculate the average results across all simulation runs
    # Group by all columns except the simulation run number and numeric metrics
    metric_columns = ['SRMSE', 'AKLD', 'MACD_S', 'MACD_C', 'runtime'] # 'AMAE',
    
    # Create the detailed results with all simulation runs
    detailed_results = combined_results.copy()
    
    # Create the averaged results
    avg_results = combined_results.groupby(['groups', 'categories_per_group', 'dependency_level', 'method'], sort=False)[metric_columns].mean().reset_index()
    
    # Calculate standard deviations
    std_results = combined_results.groupby(['groups','categories_per_group', 'dependency_level', 'method'], sort=False)[metric_columns].std().reset_index()
    
    # Rename columns to indicate they're standard deviations
    std_results.columns = [col if col in ['groups','categories_per_group', 'dependency_level', 'method'] 
                          else f"{col}_std" for col in std_results.columns]
    
    # Merge average and standard deviation results
    final_results = pd.merge(avg_results, std_results, on=['groups','categories_per_group', 'dependency_level', 'method'])
    
    return {
        'detailed_results': detailed_results,  # Contains all simulation runs
        'avg_results': avg_results,            # Just the averages
        'std_results': std_results,            # Standard deviations
        'final_results': final_results         # Averages with standard deviations
    }

def get_system_memory_info():
    """Retrieves and prints system-wide memory information."""
    mem_info = psutil.virtual_memory()
    print(f"Total RAM: {mem_info.total / (1024**3):.2f} GB")
    print(f"Used RAM: {mem_info.used / (1024**3):.2f} GB")
    print(f"Available RAM: {mem_info.available / (1024**3):.2f} GB")
    print(f"Memory Utilization: {mem_info.percent}%")

    
def print_timing_analysis(results):
    """
    Print a detailed timing analysis of the simulation results.
    """
    detailed_results = results['detailed_results']
    
    print("\n===== TIMING ANALYSIS =====")
    
    print(f"Total CPU time (sum of all method times): {detailed_results['runtime'].sum():.2f} seconds")
    
    # Method-specific timing
    print("\nAverage time per method:")
    method_times = detailed_results.groupby('method')['runtime'].mean()
    for method, time in method_times.items():
        print(f"  {method}: {time:.4f} seconds")
    
    # Group size-specific timing
    print("\nAverage time by variable groups:")
    group_times = detailed_results.groupby('groups')['runtime'].mean()
    for group, time in group_times.items():
        print(f"  {group} groups: {time:.4f} seconds")

    # Category-specific timing
    print("\nAverage time by categories per group:")
    cat_times = detailed_results.groupby('categories_per_group')['runtime'].mean()
    for cat, time in cat_times.items():
        print(f"  {cat} categories: {time:.4f} seconds")

    # Dependency level-specific timing
    print("\nAverage time by dependency level:")
    dep_times = detailed_results.groupby('dependency_level')['runtime'].mean()
    for dep, time in dep_times.items():
        print(f"  {dep}: {time:.4f} seconds")
    
    print("\n===== END TIMING ANALYSIS =====")


#----- Simulation class

class SimulationStudy:
    """
    A class to conduct simulation studies for a PhD dissertation research.
    
    Attributes:
        group_sizes (list): List of group sizes for the simulation.
        categories_per_group (list): List of categories per group for the simulation.
        rho (float): Dependency level for the simulation.
        method (str): Method used for the simulation.
        max_project (int): Maximum number of projects to be simulated.
        reference_df (pd.DataFrame): Reference DataFrame for comparison.
        target_df (pd.DataFrame): Target DataFrame for comparison.
        syn_df (pd.DataFrame): Synthetic DataFrame generated during the simulation.
    """

