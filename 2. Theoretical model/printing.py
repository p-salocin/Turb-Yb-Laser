# Created Date: Monday, April 27th 2026, 2:35:03 pm
# Author: Iván R. R. Gonzáles and Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.


def print_parameters(params, show_metrics=True):
    '''Print the model parameters in a readable format.
    Parameters:
    -----------
    params: dict
        Dictionary containing the model parameters.
    show_metrics: bool, optional
        Whether to print the fit metrics (default: True).
    '''

    print("\n" + "-" * 40)
    print("FIT METRICS:")
    print(f"  PCA Explained Variance: {params['explained_variance']:.2f}%")
    print(f"  Weighted MSE (log):     {params['mse_log_weighted']:.6f}")
    print(f"  R2 (log, weighted):     {params['r2_log_weighted']:.4f}")
    print(f"  sigma(PCA simulation):  {params['std_pca']:.4f}")
    print(f"  sigma(QML experimental):{params['std_data']:.4f}")
    print("-" * 40)