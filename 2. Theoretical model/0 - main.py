# Created Date: Tuesday, April 14th 2026, 11:00:12 am
# Author: Iván R. R. Gonzáles and Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# Import custom modules
from load_data import load_data
from optimization import optimize_parameters_for_fit
from milstein_coupled_eq import milstein_coupled_with_stochastic_variance
from fitting_metrics import calculate_fit_metrics
from plots_comp import make_histosim_comparison_scatter
from run import run_pipeline
from printing import print_parameters

# Select which dataset to analyze (CW, SML, or QML)
DATASET_NAME = 'B'  # Options: 'A' for CW and SML (dual plot), 'B' for QML

# simulation parameters for CW and SML datasets
SIM_PARAMS_A = {'N': 250000, 'dt': 1e-3,
            'I10': 0.5, 'I20': 0.5, 'nu10': 0.001, 'nu20': 0.001,
            'seed': 20, 'burn_frac': 0.3, 'xmin': -6.5, 'xmax': 6.5}

# Model parameters for CW and SML
MODEL_PARAMS_A = {
    'g11_2': 0.5, 'g22_2': 0.5,
    'g11_4': -0.055, 'g22_4': -0.055, 'g12_4': -0.055,
    'nu01': 0.003, 'nu02': 0.003,
    'gamma1': 0.65, 'gamma2': 0.65,
    'kappa1': 0.55, 'kappa2': 0.55,
}

# Bounds for CW and SML (in case of optimization)
# Disclaimer: Parameters are not being used since optimization is currently turned off. 
# Optimization function needs to be revised and restructured to work properly.
BOUNDS_A = [
        (-5.0, 5.0),        # g11_2
        (-5.0, 5.0),        # g22_2
        (-10.0, -0.1),      # g11_4
        (-10.0, -0.1),      # g22_4
        (-5.0, 0.0),        # g12_4
        (-0.1, 0.1),        # nu01
        (-0.1, 0.1),        # nu02
        (0.1, 10.0),        # gamma1
        (0.1, 10.0),        # gamma2
        (0.1, 5.0),         # kappa1
        (0.1, 5.0)          # kappa2
    ]

# simulation parameters for QML datasets
SIM_PARAMS_B = {'N': 250000, 'dt': 1e-3,
            'I10': 1.0, 'I20': 0.5, 'nu10': 0.05, 'nu20': 0.05,
            'seed': 42, 'burn_frac': 0.3, 'xmin': -10, 'xmax': 10}

# Model parameters for QML
MODEL_PARAMS_B = {   
    'g11_2': 0.3, 'g22_2': 0.3,
    'g11_4': -6.0, 'g22_4': -0.3, 'g12_4': -0.4,
    'nu01': 0.4, 'nu02': 0.4,
    'gamma1': 0.58, 'gamma2': 0.58,
    'kappa1': 4.2, 'kappa2': 4.2
}

# Bounds for QML (in case of optimization)
# Disclaimer: Parameters are not being used since optimization is currently turned off. 
# Optimization function needs to be revised and restructured to work properly.
BOUNDS_B = [
        (-5.0, 5.0),   # g11_2
        (-5.0, 5.0),   # g22_2
        (-10.0, -0.1), # g11_4
        (-10.0, -0.1), # g22_4
        (-5.0, 0.0),   # g12_4
        (1e-4, 10),    # nu01
        (1e-4, 10),    # nu02
        (0.1, 10.0),   # gamma1
        (0.1, 10.0),   # gamma2
        (0.1, 5.0),    # kappa1
        (0.1, 5.0)     # kappa2
    ]

# Start main execution
if __name__ == "__main__":

    print("=" * 60)
    print("Stochastic Coupled Oscillator - Histogram Comparison")
    print("=" * 60)

    # Define file paths for datasets
    FILE_PATH_1 = r'DATA\PCA_TIME_SERIES_FILT\CW_PC_TIME_SERIES_147_0_filtered.npy'
    FILE_PATH_2 = r'DATA\PCA_TIME_SERIES_FILT\SML_PC_TIME_SERIES_690_0_filtered.npy'
    FILE_PATH_3 = r'DATA\MODEL_DATA\QML286.mat'

    # Process datasets and generate comparison plots
    if DATASET_NAME == 'A':

        # Create a figure with two subplots for CW and SML comparisons
        fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)

        marker_size = 30 # Adjust marker size for better visibility in the scatter plots

        # First dataset
        # DO NOT RUN OPTIMIZATION. OPTIMIZATION FUNCTION NEEDS TO BE REVISED AND RESTRUCTURED TO WORK PROPERLY.  
        I1_a, I2_a, data_a, _, SIM_PARAMS = run_pipeline(FILE_PATH_1, 'npy', SIM_PARAMS_A, MODEL_PARAMS_A, BOUNDS_A, 
                                                         run_optimization=False)

        _, metrics_a = make_histosim_comparison_scatter(I1_a, I2_a, data_a, size_sct=marker_size,
                                                        label_name='CW', 
                                                        burn_frac=SIM_PARAMS['burn_frac'],
                                                        xmin=SIM_PARAMS['xmin'],
                                                        xmax=SIM_PARAMS['xmax'],
                                                        gaussian_fit=True,
                                                        ax=axes[0])

        print_parameters(metrics_a, show_metrics=False)  # Use the new function to print metrics in a formatted way

        # Second dataset
        # DO NOT RUN OPTIMIZATION. OPTIMIZATION FUNCTION NEEDS TO BE REVISED AND RESTRUCTURED TO WORK PROPERLY.  
        I1_b, I2_b, data_b, _, SIM_PARAMS = run_pipeline(FILE_PATH_2, 'npy', SIM_PARAMS_A, MODEL_PARAMS_A, BOUNDS_A, 
                                                         run_optimization=False)

        _, metrics_b = make_histosim_comparison_scatter(I1_b, I2_b, data_b, size_sct=marker_size,
                                                        label_name='SML', 
                                                        burn_frac=SIM_PARAMS['burn_frac'],
                                                        xmin=SIM_PARAMS['xmin'],
                                                        xmax=SIM_PARAMS['xmax'],
                                                        gaussian_fit=True,
                                                        ax=axes[1])

        print_parameters(metrics_b, show_metrics=False)  # Use the new function to print metrics in a formatted way

        print(f"\nGenerating and displaying histogram comparison...")

        # Set text size and font parameters
        text_size = 10
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': 'Helvetica',
            'font.size': text_size,
            'axes.linewidth': 1.05
        })

        # Add subplot labels and customize ticks
        axes[0].text(-0.10, 1.1, "A", transform=axes[0].transAxes, fontsize=text_size+2, fontweight='bold')
        axes[1].text(-0.10, 1.1, "B", transform=axes[1].transAxes, fontsize=text_size+2, fontweight='bold')
        axes[0].tick_params(axis='both', which='both', direction='in', labelsize=text_size)
        axes[1].tick_params(axis='both', which='both', direction='in', labelsize=text_size)
        axes[0].legend(loc='upper right', fontsize=text_size-1)
        axes[1].legend(loc='upper right', fontsize=text_size-1)
        
        print("Saving figure...")
        fig.savefig('FIGURES/SM_Figure_4.pdf')
        print("Figure saved as 'FIGURES/SM_Figure_4.pdf'")
    
    elif DATASET_NAME == 'B':

        # Create a single figure for QML comparison
        fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)

        marker_size = 23 # Adjust marker size for better visibility in the scatter plot

        # DO NOT RUN OPTIMIZATION. OPTIMIZATION FUNCTION NEEDS TO BE REVISED AND RESTRUCTURED TO WORK PROPERLY.
        I1, I2, data, _, SIM_PARAMS = run_pipeline(FILE_PATH_3, 'mat', SIM_PARAMS_B, MODEL_PARAMS_B, BOUNDS_B, 
                                                   run_optimization=False)
        
        # Generate scatter plot comparing histogram of simulated data with empirical data, and calculate fit metrics
        _, metrics = make_histosim_comparison_scatter(I1, I2, data, size_sct=marker_size,
                                                        label_name='QML',
                                                        burn_frac=SIM_PARAMS['burn_frac'],
                                                        xmin=SIM_PARAMS['xmin'],
                                                        xmax=SIM_PARAMS['xmax'],
                                                        gaussian_fit=False,
                                                        ax=ax)
        
        print_parameters(metrics, show_metrics=False)  # Use the new function to print metrics in a formatted way

        print(f"\nGenerating and displaying histogram comparison...")

        text_size = 8
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': 'Helvetica',
            'font.size': text_size,
            'axes.linewidth': 1.05
        })

        # Configure axis labels, ticks, and legend
        ax.tick_params(axis='both', which='both', direction='in', labelsize=text_size)
        ax.legend(loc='upper right', fontsize=text_size-1)
        
        print("Saving figure...")
        fig.savefig('FIGURES/Figure_2.pdf')
        print("Figure saved as 'FIGURES/Figure_2.pdf'")

        




