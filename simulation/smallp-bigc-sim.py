#----------------------------------------------------------
# Description: This script runs the simulations for the blockwise IPF framework 
# with standard IPF as benchmark.
# Author(s): William O. Agyapong
# Date Created: 05/15/2025
# Date Last Modified: 05/15/2025
#----------------------------------------------------------

#----- Import necessary libraries
import utils
from sim_utils import run_sim, print_timing_analysis
import timeit
import os
import multiprocessing as mp


#----- run the simulation
if __name__ == "__main__":
    # Set the random seed for reproducibility
    utils.set_seed(123)
    
    #----- Define global parameters for the simulation
    nsims =  5 # 100 # number of simulation runs
    n_cores_to_free = 3 # number of CPU cores to free for the OS
    # On the HPC, set n_cores_to_free = 0 to use all available cores
    # On a local machine, set n_cores_to_free to at least 1 to free some cores for the OS
    num_samples = 10000 # number of samples to generate. Note that only 20% of the samples are used for the reference data 
    output_path = os.path.dirname(__file__)+"/output/smallp-bigc"
    
    print(f"CPU cores available: {mp.cpu_count()}")

    #----- Case I: 6 variables, uniform number of variables and uniform categories per group -----
    # A relavely low dimensionality case to test the performance our BIPF algorithms
    # in the low dimensionality regime.
    print("\n==================== Running CASE I - C=100 =====================")

    scenarios = {
        "dependency_levels": [0.8, 0.5, 0.2], # 3 scenarios
        "group_sizes": utils.generate_groupings(p=4), # 4 variables in total, 1 scenario
        "categories_per_group": [100]*2 # for uniform categories per group for 1 scenario
    }
    
    # Calculate total number of parameter combinations
    total_combinations = len(scenarios['group_sizes']) * len(scenarios['dependency_levels']) * len(scenarios['categories_per_group']) * nsims
    print(f"Total parameter combinations to run: {total_combinations}")
    
    # Record total execution time
    total_start_time = timeit.default_timer()
    
    results1 = run_sim(scenarios,
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
    print_timing_analysis(results1)

    # Save the results to a CSV file
    results1['final_results'].to_csv(f'{output_path}/case1_4vars_100c_final_results.csv', index=False)
    # Save the detailed results to a CSV file
    results1['detailed_results'].to_csv(f'{output_path}/case1_4vars_100c_detailed_results.csv', index=False)
    
    print(f"Case I simulation completed. Results saved to {output_path}\n")


# #----- Case II: 4 variables, uniform number of variables and uniform categories per group -----
# # A relatively high dimensionality case 
#     print("\n==================== Running CASE II - C=150 =====================")
#     utils.set_seed(123)

#     scenarios = {
#         "dependency_levels": [0.8, 0.5, 0.2], # 3 scenarios
#         "group_sizes": utils.generate_groupings(p=4), # 4 variables in total, 1 scenario
#         "categories_per_group": [100]*2 # for uniform categories per group for 1 scenario
#     } 
    
#     # Calculate total number of parameter combinations
#     total_combinations = len(scenarios['group_sizes']) * len(scenarios['dependency_levels']) * len(scenarios['categories_per_group']) * nsims
#     print(f"Total parameter combinations to run: {total_combinations}")
    
#     # Record total execution time
#     total_start_time = timeit.default_timer()
    
#     results2 = run_sim(scenarios,
#                     mode='mvn',
#                     nsims=nsims,
#                     num_samples=num_samples,
#                     free_cores=n_cores_to_free
#                     )
    
#     total_end_time = timeit.default_timer()
#     total_time = total_end_time - total_start_time
#     print(f"Total execution time: {total_time:.2f} seconds")

#     print(f"Average time per parameter combination: {total_time/total_combinations:.2f} seconds")
    
#     # Printdetailed timing analysis
#     print_timing_analysis(results2)

#     # Save the results to a CSV file
#     results2['final_results'].to_csv(f'{output_path}/case2_4vars_150c_final_results.csv', index=False)
#     # Save the detailed results to a CSV file
#     results2['detailed_results'].to_csv(f'{output_path}/case2_4vars_150c_detailed_results.csv', index=False)
    
#     print(f"Case II simulation completed. Results saved to {output_path}\n")



