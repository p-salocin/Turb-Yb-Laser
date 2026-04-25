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


# simulation parameters
SIM_PARAMS = {'N': 200000, 'dt': 1e-3,
            'I10': 0.5, 'I20': 0.5, 'nu10': 0.00001, 'nu20': 0.00001,
            'seed': 42, 'burn_frac': 0.2, 'xmin': -6.5, 'xmax': 6.5}

# Model parameters for CW and SML
MODEL_PARAMS_A = {
    'g11_2': 1.0, 'g22_2': 1.0,
    'g11_4': -1.0, 'g22_4': -0.5, 'g12_4': -0.5,
    'nu01': 0.00005, 'nu02': 0.00005,
    'gamma1': .5, 'gamma2': .5,
    'kappa1': 0.4, 'kappa2': 0.4,
}

#     g11_2=1.0, g11_4=-0.7, 
#     g22_2=1.0, g22_4=-0.5, g12_4=-0.5,
#     nu01=0.00, nu02=0.00,
#     gamma1=2.0, gamma2=2.0,
#     kappa1=0.4, kappa2=0.4,

#     'g11_2': 0.03, 'g22_2': 0.03,
#     'g11_4': -0.03, 'g22_4': -0.03, 'g12_4': -0.03,
#     'nu01': 0.001, 'nu02': 0.001,
#     'gamma1': 0.5, 'gamma2': 0.5,
#     'kappa1': 0.55, 'kappa2': 0.55
#         }

BOUNDS_A = [
        (-5.0, 5.0),        # g11_2
        (-5.0, 5.0),        # g22_2
        (-10.0, -0.1),      # g11_4
        (-10.0, -0.1),      # g22_4
        (-5.0, 0.0),        # g12_4
        (-0.1, 0.1),      # nu01
        (-0.1, 0.1),      # nu02
        (0.1, 10.0),        # gamma1
        (0.1, 10.0),        # gamma2
        (0.1, 5.0),         # kappa1
        (0.1, 5.0)          # kappa2
    ]

#  I10=1.0, I20=0.5, nu10=0.001, nu20=0.001,
#     g11_2=1.0, g11_4=-0.7, g12_4=-0.5,
#     g22_2=1.0, g22_4=-0.5,
#     nu01=0.00, nu02=0.00,
#     gamma1=2.0, gamma2=2.0,
#     kappa1=0.4, kappa2=0.4,

# initial_params_A = {
#     'g11_2': 0.03, 'g22_2': 0.03,
#     'g11_4': -0.03, 'g22_4': -0.03,
#     'g12_4': -0.03,
#     'nu01': 0.001, 'nu02': 0.001,
#     'gamma1': 0.5, 'gamma2': 0.5,
#     'kappa1': 0.55, 'kappa2': 0.55
#         }











# Model parameters for QML
# MODEL_PARAMS_B = {
#     'I10': 0.5, 'I20': 0.5, 'nu10': 0.05, 'nu20': 0.05,
#     'g11_2': 1.0, 'g22_2': 1.0,
#     'g11_4': -1.0, 'g22_4': -1.0, 'g12_4': -0.5,
#     'nu01': 0.05, 'nu02': 0.05,
#     'gamma1': 2.0, 'gamma2': 2.0,
#     'kappa1': 0.4, 'kappa2': 0.4}

    # # Set default initial parameters if not provided
    # if initial_params is None:
    #     initial_params = {
    #         'g11_2': 0.3, 'g22_2': 0.3,
    #         'g11_4': -6, 'g22_4': -0.3,
    #         'g12_4': -0.4,
    #         'nu01': 0.8, 'nu02': 0.8,
    #         'gamma1': 0.5, 'gamma2': 0.5,
    #         'kappa1': 1.5, 'kappa2': 1.5
    #     }

    # # Parameter names in consistent order for vectorized optimization
    # param_names = ['g11_2', 'g22_2', 'g11_4', 'g22_4', 'g12_4',
    #                'nu01', 'nu02', 'gamma1', 'gamma2', 'kappa1', 'kappa2']
    # initial_values = [initial_params[name] for name in param_names]

    # # Define physical bounds for each parameter
    # bounds = [
    #     (-5.0, 5.0),   # g11_2: quadratic self-coupling coefficient for intensity 1
    #     (-5.0, 5.0),   # g22_2: quadratic self-coupling coefficient for intensity 2
    #     (-10.0, -0.1), # g11_4: quartic self-coupling coefficient for intensity 1
    #     (-10.0, -0.1), # g22_4: quartic self-coupling coefficient for intensity 2
    #     (-5.0, 0.0),   # g12_4: quartic cross-coupling coefficient
    #     (1e-4, 10),    # nu01: detuning parameter for intensity 1
    #     (1e-4, 10),    # nu02: detuning parameter for intensity 2
    #     (0.1, 10.0),   # gamma1: damping rate for intensity 1
    #     (0.1, 10.0),   # gamma2: damping rate for intensity 2
    #     (0.1, 5.0),    # kappa1: feedback strength for intensity 1
    #     (0.1, 5.0)     # kappa2: feedback strength for intensity 2
    # ]

# N=50_000, dt=1e-3, seed=42,
#     I10=1.0, I20=0.5, nu10=0.05, nu20=0.05,
#     g11_2=1.0, g11_4=-0.7, g12_4=-0.5,
#     g22_2=1.0, g22_4=-0.5,
#     nu01=0.05, nu02=0.05,
#     gamma1=2.0, gamma2=2.0,
#     kappa1=0.4, kappa2=0.4,
#     clamp_nonneg=True):


if __name__ == "__main__":

    print("=" * 60)
    print("Stochastic Coupled Oscillator - Histogram Comparison")
    print("=" * 60)

    FILE_PATH_1 = r'DATA\PCA_TIME_SERIES_FILT\CW_PC_TIME_SERIES_147_0_filtered.npy'
    FILE_PATH_2 = r'DATA\PCA_TIME_SERIES_FILT\SML_PC_TIME_SERIES_690_0_filtered.npy'

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)

    marker_size = 30

    # First dataset
    I1_a, I2_a, data_a, _, SIM_PARAMS = run_pipeline(FILE_PATH_1, 'npy', SIM_PARAMS, MODEL_PARAMS_A, BOUNDS_A, run_optimization=True)

    _, metrics_a = make_histosim_comparison_scatter(
        I1_a, I2_a, data_a,
        size_sct=marker_size,
        label_name='CW', 
        burn_frac=SIM_PARAMS['burn_frac'],
        xmin=SIM_PARAMS['xmin'],
        xmax=SIM_PARAMS['xmax'],
        gaussian_fit=True,
        ax=axes[0]
        )

    print("\n" + "-" * 40)
    print("FIT METRICS:")
    print(f"  PCA Explained Variance: {metrics_a['explained_variance']:.2f}%")
    print(f"  Weighted MSE (log):     {metrics_a['mse_log_weighted']:.6f}")
    print(f"  R2 (log, weighted):     {metrics_a['r2_log_weighted']:.4f}")
    print(f"  sigma(PCA simulation):  {metrics_a['std_pca']:.4f}")
    print(f"  sigma(QML experimental):{metrics_a['std_data']:.4f}")
    print("-" * 40)

    # Second dataset
    I1_b, I2_b, data_b, _, SIM_PARAMS = run_pipeline(FILE_PATH_2, 'npy', SIM_PARAMS, MODEL_PARAMS_A, BOUNDS_A, run_optimization=True)

    _, metrics_b = make_histosim_comparison_scatter(
        I1_b, I2_b, data_b,
        size_sct=marker_size,
        label_name='SML', 
        burn_frac=SIM_PARAMS['burn_frac'],
        xmin=SIM_PARAMS['xmin'],
        xmax=SIM_PARAMS['xmax'],
        gaussian_fit=True,
        ax=axes[1]
    )

    print("\n" + "-" * 40)
    print("FIT METRICS:")
    print(f"  PCA Explained Variance: {metrics_b['explained_variance']:.2f}%")
    print(f"  Weighted MSE (log):     {metrics_b['mse_log_weighted']:.6f}")
    print(f"  R2 (log, weighted):     {metrics_b['r2_log_weighted']:.4f}")
    print(f"  sigma(PCA simulation):  {metrics_b['std_pca']:.4f}")
    print(f"  sigma(QML experimental):{metrics_b['std_data']:.4f}")
    print("-" * 40)

    print(f"\nGenerating and displaying histogram comparison...")

    # Set text size and font parameters
    text_size = 10
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': 'Helvetica',
        'font.size': text_size,
        'axes.linewidth': 1.05
    })

    axes[0].text(-0.10, 1.1, "A", transform=axes[0].transAxes, fontsize=text_size+2, fontweight='bold')
    axes[1].text(-0.10, 1.1, "B", transform=axes[1].transAxes, fontsize=text_size+2, fontweight='bold')
    axes[0].tick_params(axis='both', which='both', direction='in', labelsize=text_size)
    axes[1].tick_params(axis='both', which='both', direction='in', labelsize=text_size)

    axes[0].legend(loc='upper right', fontsize=text_size-1)
    axes[1].legend(loc='upper right', fontsize=text_size-1)
    
    # print("Saving figure...")
    # fig.savefig('FIGURES/SM_Figure_4.pdf')
    plt.show()

