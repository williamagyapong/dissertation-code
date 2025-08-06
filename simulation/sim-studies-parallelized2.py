#----------------------------------------------------------
# Description: This script runs the simulations for the blockwise IPF framework 
# with standard IPF as benchmark.
# Author(s): William O. Agyapong
# Date Created: 04/02/2025
# Date Last Modified: 04/05/2025
#----------------------------------------------------------

# Import required libraries
import numpy as np
import pandas as pd
import timeit
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
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
                         reference_df, target_df, syn_df, start_time, stop_time, error=None):
    
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
        "time_taken (s)": stop_time - start_time
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

    #-- Apply algorithms
    # Standard IPF
    if run_ipf:
        print(f"Running standard IPF for simulation {sim + 1} with groups {group_sizes} and categories {categories_per_group} and rho {rho}\n")
        try:
            start_time = timeit.default_timer()
            std_ipf_syn, _ = ipf_utils.std_ipf(ref_set=reference_df, target_set=target_df,
                                                conv_rate=tol,selected_features=selected_features)
            stop_time = timeit.default_timer()
            results.append(
                package_sim_results(sim, group_sizes, categories_per_group, rho, "IPF", max_project, reference_df, target_df, std_ipf_syn, start_time, stop_time)
            )
        except Exception as e:
            print(f"Error in standard IPF: {e}")
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
                "time_taken (s)": np.nan
            }
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
                "time_taken (s)": np.nan
            }
        )

    # # Apply blockwise IPF on true feature groups
    # target_margins_dict = ipf_utils.ipf_target_in(target_df, reference_df, target_df.columns)
    
    # # N-BIPF
    # feature_seq = utils.get_variable_names_by_group(selected_features, group_sizes) # determine the feature sequence
    # start_time = timeit.default_timer()
    # nbipf_syn, _ = ipf_utils.nbipf(reference_df, target_margins_dict, feature_seq, ipf_utils.ipfn_wrapper, ipf_args={'convergence_rate':tol}, n=target_df.shape[0], runs=1)
    # stop_time = timeit.default_timer()
    # results.append(
    #     package_sim_results(sim, group_sizes, categories_per_group, rho, "N-BIPF", max_project, reference_df, target_df, nbipf_syn, start_time, stop_time)
    # )

    # # S-BIPF
    # feature_seq = utils.get_variable_names_by_group(selected_features, group_sizes)
    # start_time = timeit.default_timer()
    # sbipf_syn = ipf_utils.sbipf(reference_df, target_margins_dict, feature_seq, ipf_utils.ipfn_wrapper, ipf_args={'convergence_rate':tol}, n=target_df.shape[0], runs=nruns)
    # stop_time = timeit.default_timer()
    # results.append(
    #     package_sim_results(sim, group_sizes, categories_per_group, rho, "S-BIPF", max_project, reference_df, target_df, sbipf_syn, start_time, stop_time)
    # )

    # # N-BIPF-RP
    # start_time = timeit.default_timer()
    # nbipf_rp_syn, _ = ipf_utils.nbipf(reference_df, target_margins_dict, feature_seq, ipf_utils.ipfn_wrapper, ipf_args={'convergence_rate':tol}, n=target_df.shape[0], runs=1)
    # stop_time = timeit.default_timer()
    # results.append(
    #     package_sim_results(sim, group_sizes, categories_per_group, rho, "N-BIPF-RP",
    #                          max_project, reference_df, target_df, nbipf_rp_syn, start_time, stop_time)
    # )

    # # S-BIPF-RP
    # feature_seq = ipf_utils.group_features(selected_features, n_group=len(group_sizes), random_seed=123)
    # start_time = timeit.default_timer()
    # sbipf_rp_syn = ipf_utils.sbipf(reference_df, target_margins_dict, feature_seq, ipf_utils.ipfn_wrapper, ipf_args={'convergence_rate':tol}, n=target_df.shape[0], runs=nruns)
    # stop_time = timeit.default_timer()
    # results.append(
    #     package_sim_results(sim, group_sizes, categories_per_group, rho, "S-BIPF-RP",
    #                          max_project, reference_df, target_df, sbipf_rp_syn, start_time, stop_time)
    # )
    
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
    num_cores_to_use = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count())) # use SLURM_CPUS_PER_TASK if available, otherwise use all available cores
    if num_cores_to_use == num_cores: # if not running on a cluster
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
    # with ProcessPoolExecutor(max_workers=num_cores_to_use) as executor:
    #     # Map the function to all parameter combinations
    #     for result in tqdm(executor.map(run_param_partial, all_combinations), total=len(all_combinations), desc="Processing parameter combinations"):
    #         all_results.extend(result)
    
    with ProcessPoolExecutor(max_workers=num_cores_to_use) as executor:
        # Submit all tasks
        future_to_params = {executor.submit(run_param_partial, params): params for params in all_combinations}
        
        # Process results as they complete with a progress bar
        # with tqdm(total=len(all_combinations), desc="Processing parameter combinations") as pbar:
        for future in tqdm(as_completed(future_to_params), total=len(all_combinations), desc="Processing parameter combinations"):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"Error processing {future_to_params[future]}: {e}")
            
            # pbar.update(1)
    
    # Convert all results to a DataFrame
    combined_results = pd.DataFrame(all_results)
    
    # Calculate the average results across all simulation runs
    # Group by all columns except the simulation run number and numeric metrics
    metric_columns = ['SRMSE 1', 'AMAE', 'AKLD', 'MACD_S', 'MACD_C', 'time_taken (s)']
    
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


def print_timing_analysis(results):
    """
    Print a detailed timing analysis of the simulation results.
    """
    detailed_results = results['detailed_results']
    
    print("\n===== TIMING ANALYSIS =====")
    
    print(f"Total CPU time (sum of all method times): {detailed_results['time_taken (s)'].sum():.2f} seconds")
    
    # Method-specific timing
    print("\nAverage time per method:")
    method_times = detailed_results.groupby('method')['time_taken (s)'].mean()
    for method, time in method_times.items():
        print(f"  {method}: {time:.4f} seconds")
    
    # Group size-specific timing
    print("\nAverage time by variable groups:")
    group_times = detailed_results.groupby('groups')['time_taken (s)'].mean()
    for group, time in group_times.items():
        print(f"  {group} groups: {time:.4f} seconds")

    # Category-specific timing
    print("\nAverage time by categories per group:")
    cat_times = detailed_results.groupby('categories_per_group')['time_taken (s)'].mean()
    for cat, time in cat_times.items():
        print(f"  {cat} categories: {time:.4f} seconds")

    # Dependency level-specific timing
    print("\nAverage time by dependency level:")
    dep_times = detailed_results.groupby('dependency_level')['time_taken (s)'].mean()
    for dep, time in dep_times.items():
        print(f"  {dep}: {time:.4f} seconds")
    
    print("\n===== END TIMING ANALYSIS =====")


#----- run the simulation
if __name__ == "__main__":
    # Set the random seed for reproducibility
    utils.set_seed(123)
    
    #----- Define global parameters for the simulation
    nsims =  1 # 100 # number of simulation runs
    n_cores_to_free = 0 # number of CPU cores to free for the OS
    # On the HPC, set n_cores_to_free = 0 to use all available cores
    # On a local machine, set n_cores_to_free to at least 1 to free some cores for the OS
    num_samples = 10000 # number of samples to generate. Note that only 20% of the samples are used for the reference data 
    output_path = os.path.dirname(__file__)+"/output/"
    
    print(f"CPU cores available: {mp.cpu_count()}")


#----- Case III: 16 variables, uniform number of variables and uniform categories per group -----
# A high dimensionality case 
    print("\n==================== Running CASE II =====================")
    utils.set_seed(123)

    scenarios = {
        "dependency_levels": [0.8, 0.5, 0.2], # 3 scenarios
        "group_sizes": utils.generate_groupings(p=12), # 16 variables in total, 3 scenarios
        "categories_per_group": [2, 4, 6] # for uniform categories per group for 3 scenarios
    } 
    
    # Calculate total number of parameter combinations
    total_combinations = len(scenarios['group_sizes']) * len(scenarios['dependency_levels']) * len(scenarios['categories_per_group']) * nsims
    print(f"Total parameter combinations to run: {total_combinations}")
    
    # Record total execution time
    total_start_time = timeit.default_timer()
    
    results3 = run_sim(scenarios,
                    mode='mvn',
                    nsims=nsims,
                    num_samples=num_samples,
                    free_cores=n_cores_to_free
                    )
    
    total_end_time = timeit.default_timer()
    total_time = total_end_time - total_start_time
    print(f"Total execution time: {total_time:.2f} seconds")

    print(f"Average time per parameter combination: {total_time/total_combinations:.2f} seconds")
    
    # Printdetailed timing analysis
    print_timing_analysis(results3)

    # Save the results to a CSV file
    # results3['final_results'].to_csv(f'{output_path}/case3_16vars_final_results.csv', index=False)
    # # Save the detailed results to a CSV file
    # results3['detailed_results'].to_csv(f'{output_path}/case3_16vars_detailed_results.csv', index=False)

    results3['final_results'].to_csv(f'{output_path}/test_case_final_results.csv', index=False)
    # Save the detailed results to a CSV file
    results3['detailed_results'].to_csv(f'{output_path}/test_case_detailed_results.csv', index=False)
    
    print(f"Case III simulation completed. Results saved to {output_path}\n")



    #----- Case II: 16 variables, and uniform number of variables and categories per group -----

    # print("\n==================== Running CASE II =====================")
    # scenarios = {
    #     "dependency_levels": [0.8, 0.5, 0.2], # 3 scenarios
    #     "group_sizes": utils.generate_groupings(p=16), # 16 variables in total, 3 scenarios
    #     "categories_per_group": [2, 4, 6] # for uniform categories per group for 3 scenarios
    # }
    
    # # Calculate total number of parameter combinations
    # total_combinations = len(scenarios['group_sizes']) * len(scenarios['dependency_levels']) * len(scenarios['categories_per_group']) * nsims
    # print(f"Total parameter combinations to run: {total_combinations}")
    
    # # Record total execution time
    # total_start_time = timeit.default_timer()
    
    # results2 = run_sim(scenarios,
    #                 mode='mvn',
    #                 nsims=nsims,
    #                 num_samples=num_samples,
    #                 free_cores=6
    #                 )
    
    # total_end_time = timeit.default_timer()
    # total_time = total_end_time - total_start_time
    # print(f"Total execution time: {total_time:.2f} seconds")

    # print(f"Average time per parameter combination: {total_time/total_combinations:.2f} seconds")
    
    # # Printdetailed timing analysis
    # print_timing_analysis(results2)

    # # Save the results to a CSV file
    # results2['final_results'].to_csv('sim-out/case2_16vars_final_results.csv', index=False)
    # # Save the detailed results to a CSV file
    # results2['detailed_results'].to_csv('sim-out/case2_16vars_detailed_results.csv', index=False)
    
    # print("Results saved to sim-out/case2_16vars_final_results.csv and sim-out/case2_16vars_detailed_results.csv")
