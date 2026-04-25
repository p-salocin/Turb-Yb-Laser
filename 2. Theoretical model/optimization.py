# Created Date: Tuesday, April 14th 2026, 3:30:11 pm
# Author: Iván R. R. Gonzáles

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.


# import necessary libraries
from scipy.optimize import minimize
from milstein_coupled_eq import milstein_coupled_with_stochastic_variance
from fitting_metrics import calculate_fit_metrics


def optimize_parameters_for_fit(loading_data, sim_params, initial_params, bounds):
    """
    Optimize coupled fiber laser model parameters to fit experimental data.
    
    This function uses L-BFGS-B optimization to tune 15 physical parameters
    (I10, I20, nu10, nu20, g11_2, g22_2, g11_4, g22_4, g12_4, nu01, nu02, gamma1, gamma2, kappa1, kappa2)
    by minimizing the weighted mean squared error between simulated intensity
    trajectories and experimental data.
    
    Parameters:
    -----------
    loading_data : array-like
        Experimental data to fit against
    sim_params : dict
        Simulation parameters (N, dt, seed, etc.)
    initial_params : dict, optional
        Initial parameter guess. If None, a default set is used.
        Keys: 'I10', 'I20', 'nu10', 'nu20', 'g11_2', 'g22_2', 'g11_4', 'g22_4', 'g12_4', 'nu01', 'nu02',
              'gamma1', 'gamma2', 'kappa1', 'kappa2'
    bounds : list of tuples
        Bounds for each parameter in the same order as initial_params.
    
    Returns:
    --------
    tuple : (optimized_params, best_mse, success)
        optimized_params (dict): Best parameter values found
        best_mse (float): Minimum weighted log MSE achieved
        success (bool): Whether optimization converged successfully
    """

    # Define parameter names and initial values
    param_names = ['g11_2', 'g22_2', 'g11_4', 'g22_4', 'g12_4',
                   'nu01', 'nu02', 'gamma1', 'gamma2', 'kappa1', 'kappa2']
    
    # Create initial parameter vector from dictionary
    initial_values = [initial_params[name] for name in param_names]

    def objective_function(params):
        """
        Calculate fit quality metric for given parameters.
        
        Runs stochastic Milstein simulation and compares against experimental data
        using weighted logarithmic mean squared error.
        Returns large penalty (1e6) if simulation fails.
        """
        # Convert parameter vector to named dictionary
        param_dict = dict(zip(param_names, params))
        try:
            # Run coupled stochastic differential equations simulation
            I1, I2, _, _, _ = milstein_coupled_with_stochastic_variance(
                N=sim_params['N'], dt=sim_params['dt'], seed=sim_params['seed'],
                I10=sim_params['I10'], I20=sim_params['I20'], nu10=sim_params['nu10'], nu20=sim_params['nu20'],
                **param_dict)
            
            # Calculate fit metrics against experimental data
            info = calculate_fit_metrics(I1, I2, loading_data,
                                         burn_frac=sim_params['burn_frac'],
                                         xmin=sim_params['xmin'], xmax=sim_params['xmax'])
            
            return info['mse_log_weighted']
        except Exception:
            # Return penalty value if simulation fails
            return 1e6

    # Perform L-BFGS-B optimization with bounds
    print("Starting parameter optimization...")
    result = minimize(objective_function, initial_values,
                     bounds=bounds, method='L-BFGS-B',
                     options={'maxiter': 30, 'ftol': 1e-4})

    # Extract and format results
    optimized_params = dict(zip(param_names, result.x))
    best_mse = result.fun
    success = result.success

    # Report results
    print(f"Optimization finished. Success: {success}")
    print(f"Best Weighted MSE (log): {best_mse:.6f}\n")

    
    return optimized_params, success