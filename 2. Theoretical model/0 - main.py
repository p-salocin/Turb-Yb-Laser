

import matplotlib.pyplot as plt
import numpy as np

from load_data import load_data
from optimization import optimize_parameters_for_qml_fit
from milstein_coupled_eq import milstein_coupled_with_stochastic_variance
from fitting_metrics import calculate_fit_metrics
from plots_comp import make_histosim_comparison_scatter


QML_FILE_PATH = r'DATA/MODEL_DATA/QML286.mat'

# simulation parameters
SIM_PARAMS = {'N': 50000, 'dt': 1e-3, 'seed': 42, 'burn_frac': 0.3, 'ymin': -10, 'ymax': 10}

MODEL_PARAMS = {
    'I10': 0.5, 'I20': 0.5, 'nu10': 0.05, 'nu20': 0.05,
    'g11_2': 1.0, 'g22_2': 1.0,
    'g11_4': -1.0, 'g22_4': -1.0, 'g12_4': -0.5,
    'nu01': 0.05, 'nu02': 0.05,
    'gamma1': 2.0, 'gamma2': 2.0,
    'kappa1': 0.4, 'kappa2': 0.4}


def main(run_optimization=True):
    plt.close('all')
    print("=" * 60)
    print("Stochastic Coupled Oscillator - Histogram Comparison")
    print("=" * 60)

    print(f"\n[1/4] Loading QML data from: {QML_FILE_PATH}")
    QML_data = load_data(QML_FILE_PATH)

    if run_optimization:
        print(f"\n[2/4] Optimizing parameters to match QML distribution...")
        opt_params, best_mse, opt_success = optimize_parameters_for_qml_fit(
            QML_data, ymin=SIM_PARAMS['ymin'], ymax=SIM_PARAMS['ymax']
        )
        print("Optimized Parameters:")
        for k, v in opt_params.items():
            print(f"  {k:8s}: {v:.4f}")
        if not opt_success:
            print("\nOptimization did not converge. Falling back to default MODEL_PARAMS.")
            opt_params = MODEL_PARAMS.copy()
    else:
        print(f"\n[2/4] Skipping optimization. Using default MODEL_PARAMS.")
        opt_params = MODEL_PARAMS.copy()

    print(f"\n[3/4] Running final Milstein simulation (N={SIM_PARAMS['N']}, dt={SIM_PARAMS['dt']})")
    I1, I2, nu1, nu2, used_params = milstein_coupled_with_stochastic_variance(
        N=SIM_PARAMS['N'],
        dt=SIM_PARAMS['dt'],
        seed=SIM_PARAMS['seed'],
        I10=0.5, I20=0.5, nu10=0.05, nu20=0.05,
        **opt_params
    )
    print("Simulation completed")

    print(f"\n[4/4] Generating and displaying histogram comparison...")
    fig, metrics = make_histosim_comparison_scatter(
        I1, I2, QML_data,
        size_sct=23,
        show=False,
        burn_frac=SIM_PARAMS['burn_frac'],
        ymin=SIM_PARAMS['ymin'],
        ymax=SIM_PARAMS['ymax'],

        )

    print("\n" + "-" * 40)
    print("FIT METRICS:")
    print(f"  PCA Explained Variance: {metrics['explained_variance']:.2f}%")
    print(f"  Weighted MSE (log):     {metrics['mse_log_weighted']:.6f}")
    print(f"  R2 (log, weighted):     {metrics['r2_log_weighted']:.4f}")
    print(f"  sigma(PCA simulation):  {metrics['std_pca']:.4f}")
    print(f"  sigma(QML experimental):{metrics['std_qml']:.4f}")
    print("-" * 40)

    return metrics, opt_params if run_optimization else None

if __name__ == "__main__":

    metrics, opt_params = main(run_optimization=True)

