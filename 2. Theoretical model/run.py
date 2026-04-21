# Created Date: Thursday, April 16th 2026, 12:49:49 pm
# Author: Iván R. R. Gonzáles and Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
from load_data import load_data
from optimization import optimize_parameters_for_fit
from milstein_coupled_eq import milstein_coupled_with_stochastic_variance


def run_pipeline(file_path, type_archive, SIM_PARAMS, MODEL_PARAMS, run_optimization=True):
    """
    Run the complete pipeline for simulating and comparing Yb fiber laser turbulence data.

    This function loads experimental data, optionally optimizes model parameters to fit the data,
    and runs a Milstein stochastic simulation using the optimized or default parameters.

    Parameters:
    file_path (str): Path to the experimental data file.
    type_archive (str): Type of the data archive ('mat' for MATLAB, otherwise NumPy).
    SIM_PARAMS (dict): Dictionary containing simulation parameters (e.g., N, dt, seed, xmin, xmax).
    MODEL_PARAMS (dict): Dictionary of default model parameters for the simulation.
    run_optimization (bool, optional): Whether to perform parameter optimization. Default is True.

    Returns:
    I1 (array): Simulated intensity array for the first component.
    I2 (array): Simulated intensity array for the second component.
    data (array): Loaded experimental data.
    used_params (dict): Parameters used in the simulation (optimized or default).
    SIM_PARAMS (dict): Original simulation parameters.
    """

    # Step 1: Load experimental data
    print(f"\n[1/3] Loading experimental data from: {file_path}")
    data = load_data(file_path, type_archive)

    # Step 2: Optimize parameters or use defaults
    if run_optimization:
        print(f"\n[2/3] Optimizing parameters to match QML distribution...")
        opt_params, best_mse, opt_success = optimize_parameters_for_fit(
            data, xmin=SIM_PARAMS['xmin'], xmax=SIM_PARAMS['xmax']
        )
        print("Optimized Parameters:")
        for k, v in opt_params.items():
            print(f"  {k:8s}: {v:.4f}")
        if not opt_success:
            print("\nOptimization did not converge. Falling back to default MODEL_PARAMS.")
            opt_params = MODEL_PARAMS.copy()
    else:
        print(f"\n[2/3] Skipping optimization. Using default MODEL_PARAMS.")
        opt_params = MODEL_PARAMS.copy()

    # Step 3: Run the Milstein simulation with the chosen parameters
    print(f"\n[3/3] Running final Milstein simulation (N={SIM_PARAMS['N']}, dt={SIM_PARAMS['dt']})")
    I1, I2, nu1, nu2, used_params = milstein_coupled_with_stochastic_variance(
        N=SIM_PARAMS['N'],
        dt=SIM_PARAMS['dt'],
        seed=SIM_PARAMS['seed'],
        I10=0.5, I20=0.5, nu10=0.05, nu20=0.05,
        **opt_params
    )
    print(f"Simulated intensities: I1={I1[-1]:.4f}, I2={I2[-1]:.4f}")

    # Return simulation results and data
    return I1, I2, data, used_params, SIM_PARAMS