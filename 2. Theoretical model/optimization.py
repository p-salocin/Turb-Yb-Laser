# Created Date: Tuesday, April 14th 2026, 3:30:11 pm
# Author: Iván R. R. Gonzáles

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.


# import necessary libraries
from scipy.optimize import minimize
from milstein_coupled_eq import milstein_coupled_with_stochastic_variance
from fitting_metrics import calculate_fit_metrics


def optimize_parameters_for_qml_fit(QML_data, initial_params=None, ymin=-10, ymax=10):
    """
    Optimize coupled fiber laser model parameters to fit experimental QML data.
    
    This function uses L-BFGS-B optimization to tune 11 physical parameters
    (g11_2, g22_2, g11_4, g22_4, g12_4, nu01, nu02, gamma1, gamma2, kappa1, kappa2)
    by minimizing the weighted mean squared error between simulated intensity
    trajectories and experimental quantum-limited measurements.
    
    Parameters:
    -----------
    QML_data : array-like
        Experimental quantum-limited measurement data to fit against
    initial_params : dict, optional
        Initial parameter guess. If None, a default set is used.
        Keys: 'g11_2', 'g22_2', 'g11_4', 'g22_4', 'g12_4', 'nu01', 'nu02',
              'gamma1', 'gamma2', 'kappa1', 'kappa2'
    ymin : float, optional
        Lower bound for data windowing in fit metrics (default: -10)
    ymax : float, optional
        Upper bound for data windowing in fit metrics (default: 10)
    
    Returns:
    --------
    tuple : (optimized_params, best_mse, success)
        optimized_params (dict): Best parameter values found
        best_mse (float): Minimum weighted log MSE achieved
        success (bool): Whether optimization converged successfully
    """
    # Set default initial parameters if not provided
    if initial_params is None:
        initial_params = {
            'g11_2': 0.3, 'g22_2': 0.3,
            'g11_4': -6, 'g22_4': -0.3,
            'g12_4': -0.4,
            'nu01': 0.8, 'nu02': 0.8,
            'gamma1': 0.5, 'gamma2': 0.5,
            'kappa1': 1.5, 'kappa2': 1.5
        }

    # Parameter names in consistent order for vectorized optimization
    param_names = ['g11_2', 'g22_2', 'g11_4', 'g22_4', 'g12_4',
                   'nu01', 'nu02', 'gamma1', 'gamma2', 'kappa1', 'kappa2']
    initial_values = [initial_params[name] for name in param_names]

    # Define physical bounds for each parameter
    bounds = [
        (-5.0, 5.0),   # g11_2: quadratic self-coupling coefficient for intensity 1
        (-5.0, 5.0),   # g22_2: quadratic self-coupling coefficient for intensity 2
        (-10.0, -0.1), # g11_4: quartic self-coupling coefficient for intensity 1
        (-10.0, -0.1), # g22_4: quartic self-coupling coefficient for intensity 2
        (-5.0, 0.0),   # g12_4: quartic cross-coupling coefficient
        (1e-4, 10),    # nu01: detuning parameter for intensity 1
        (1e-4, 10),    # nu02: detuning parameter for intensity 2
        (0.1, 10.0),   # gamma1: damping rate for intensity 1
        (0.1, 10.0),   # gamma2: damping rate for intensity 2
        (0.1, 5.0),    # kappa1: feedback strength for intensity 1
        (0.1, 5.0)     # kappa2: feedback strength for intensity 2
    ]

    def objective_function(params):
        """
        Calculate fit quality metric for given parameters.
        
        Runs stochastic Milstein simulation and compares against QML data
        using weighted logarithmic mean squared error.
        Returns large penalty (1e6) if simulation fails.
        """
        # Convert parameter vector to named dictionary
        param_dict = dict(zip(param_names, params))
        try:
            # Run coupled stochastic differential equations simulation
            I1, I2, _, _, _ = milstein_coupled_with_stochastic_variance(
                N=30000, dt=1e-3, seed=42,
                I10=0.5, I20=0.5,
                **param_dict
            )
            # Calculate fit metrics against experimental data
            info = calculate_fit_metrics(I1, I2, QML_data,
                                         burn_frac=0.3,
                                         ymin=ymin, ymax=ymax)
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
    return optimized_params, best_mse, success