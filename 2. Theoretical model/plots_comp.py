# Created Date: Tuesday, April 14th 2026, 4:41:19 pm
# Author: Iván R. R. Gonzáles and Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.


# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Import custom modules
from fitting_metrics import calculate_fit_metrics


def make_histosim_comparison_scatter(I1, I2, loading_data, size_sct, label_name,
                                     gaussian_fit=False, ax=None, burn_frac=0.2, xmin=-10, xmax=10):
    """
    Create a scatter plot comparing histograms of simulated and experimental data.

    Parameters:
    I1 (array-like): First input signal array for simulation.
    I2 (array-like): Second input signal array for simulation.
    loading_data (array-like): Experimental data for comparison.
    size_sct (int): Size of scatter points.
    gaussian_fit (bool, optional): Whether to overlay a Gaussian fit. Default is False.
    ax (matplotlib.axes.Axes, optional): Existing axis to plot on. If None, creates a new figure.
    burn_frac (float, optional): Fraction of data to burn-in (ignore) for metrics. Default is 0.2.
    xmin (float, optional): Minimum x-value for histogram. Default is -10.
    xmax (float, optional): Maximum x-value for histogram. Default is 10.

    Returns:
    fig (matplotlib.figure.Figure): The figure object containing the plot.
    metrics (dict): Dictionary of fit metrics, excluding histogram data.
    """
    # Calculate fit metrics and extract histograms
    metrics = calculate_fit_metrics(I1, I2, loading_data, burn_frac, xmin, xmax)
    H_sim = metrics['H_sim']
    H_data = metrics['H_data']

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    else:
        fig = ax.figure
    
    # Plot experimental data as blue scatter points
    ax.scatter(H_data[:, 0], H_data[:, 1], color='#0000FF', alpha=0.5, s=size_sct,
               edgecolors='navy', zorder=3, label=f'{label_name}')
    # Plot simulated data as red square scatter points
    ax.scatter(H_sim[:, 0], H_sim[:, 1], color='#FF0000', alpha=0.4, s=size_sct,
               edgecolors='darkred', marker='s', zorder=3, label='Model')
    
    # Optionally plot Gaussian fit
    if gaussian_fit == True:
        x_gauss = np.linspace(xmin, xmax, 500)
        gauss_pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_gauss**2)
        ax.plot(x_gauss, gauss_pdf, color='gray', linewidth=2, zorder=1, label='Gaussian')

    # Set axis labels, ticks, and scales
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$P(x)$')
    ax.set_yscale('log')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(1e-5, 5)

    # Remove histogram data from metrics to avoid redundancy
    metrics.pop('H_sim', None)
    metrics.pop('H_data', None)

    # Return the figure and metrics
    return fig, metrics
