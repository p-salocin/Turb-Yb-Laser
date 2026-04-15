# Created Date: Friday, March 27th 2026, 2:22:02 pm
# Authors: Iván R. R. Gonzáles and Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original authors.


# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from ts2vg import NaturalVG
from scipy.optimize import curve_fit

# Import custom modules
from power_law import power_law
from degree_distr import compute_degree_distribution

# Paths to data files
files = [
    "DATA/PCA_TIME_SERIES_FILT/CW_PC_TIME_SERIES_147_0_filtered.npy",
    "DATA/PCA_TIME_SERIES_FILT/QML_PC_TIME_SERIES_286_0_filtered.npy",
    "DATA/PCA_TIME_SERIES_FILT/SML_PC_TIME_SERIES_690_0_filtered.npy"]

# Load the time series
series_a = np.load(files[0])
series_b = np.load(files[1])
series_c = np.load(files[2])

# Intervals for fitting power-law distributions
interval =[(10, 250), (7, 35), (601, 3338)] 

# Create figure
fig = plt.figure(figsize=(3.5, 4.6), constrained_layout=True)

# Set global plot style
text_size = 8
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': text_size,
    'axes.linewidth': 1.05,
})

# Colors for plotting
colors = ['#0000FF', "#E63946"]

# Create gridspec for 3 rows and 1 column
gs = fig.add_gridspec(3, 1, height_ratios=[0.6, 1, 0.6])

# Add subplots for each row
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[1, 0])
ax_c = fig.add_subplot(gs[2, 0])


# ==== Plot a ====
# Compute degree distribution and fit power-law for series_a
ks_a, ps_a, ks_valid_a, gamma_a, C_a = compute_degree_distribution(series_a, k_min=interval[0][0], k_max=interval[0][1])

# Plot degree distribution and fitted power-law for series_a
ax_a.scatter(ks_a, ps_a, s=7.2, facecolors='none', edgecolors=colors[0], linewidth=0.6, alpha=0.5)

# Fit power-law and plot the fitted curve for series_a
ks_fit = np.logspace(np.log10(ks_valid_a.min()), np.log10(ks_valid_a.max()), 200)

# Plot the fitted power-law curve for series_a
ax_a.plot(ks_fit, power_law(ks_fit, gamma_a, C_a), '--', color=colors[1], lw=2)

# Add text labels and annotation for the fitted exponent gamma_a
ax_a.set_xscale("log"); ax_a.set_yscale("log")
ax_a.set_xlabel(r"$k$")
ax_a.set_ylabel(r"$p(k)$",  labelpad=12)
ax_a.text(0.25, 0.55, rf"$\gamma = {gamma_a:.2f}$", transform=ax_a.transAxes, fontsize=text_size, color=colors[1])
ax_a.tick_params(axis='both', which='both', direction='in', top=True, right=True)

# ==== Plot b ====
# Compute degree distribution and fit power-law for series_b in two intervals
ks_b1, ps_b1, ks_valid_b1, gamma_b1, C_b1 = compute_degree_distribution(series_b, k_min=interval[1][0], k_max=interval[1][1])
ks_b2, ps_b2, ks_valid_b2, gamma_b2, C_b2 = compute_degree_distribution(series_b, k_min=interval[2][0], k_max=interval[2][1])

# Plot degree distribution and fitted power-law for series_b
ax_b.scatter(ks_b1, ps_b1, s=7.2, facecolors='none', edgecolors=colors[0], linewidth=0.6, alpha=0.5)

# Plot the fitted power-law curves for series_b
ks_fit_b1 = np.logspace(np.log10(ks_valid_b1.min()), np.log10(ks_valid_b1.max()), 200)
ks_fit_b2 = np.logspace(np.log10(ks_valid_b2.min()), np.log10(ks_valid_b2.max()), 200)

# Plot the fitted power-law curves for series_b
ax_b.plot(ks_fit_b1, power_law(ks_fit_b1, gamma_b1, C_b1), '--', color=colors[1], lw=2)
ax_b.plot(ks_fit_b2, power_law(ks_fit_b2, gamma_b2, C_b2), '--', color=colors[1], lw=2)

# Add text labels and annotation for the fitted exponents gamma_b1 and gamma_b2
ax_b.text(0.10, 0.55, rf"$\gamma_1 = {gamma_b1:.2f}$", transform=ax_b.transAxes, fontsize=text_size, color=colors[1])
ax_b.text(0.45, 0.25, rf"$\gamma_2 = {gamma_b2:.2f}$", transform=ax_b.transAxes, fontsize=text_size, color=colors[1])
ax_b.set_xscale("log"); ax_b.set_yscale("log")
ax_b.set_xlabel(r"$k$")
ax_b.set_ylabel(r"$p(k)$", labelpad=12)
ax_b.tick_params(axis='both', which='both', direction='in', top=True, right=True)
ax_b.set_xlim(1.5e0)

# --- Inset inside plot b (top-left) ---
ax_inset = ax_b.inset_axes([0.635, 0.56, 0.35, 0.4])  # top-left

# Shuffle series_b to create a randomized version and compute degree distribution for the shuffled series
shuffled_series = series_b.copy()

# Shuffle the series multiple times to ensure a good randomization
for _ in range(3):
    np.random.shuffle(shuffled_series)

# Compute degree distribution and fit power-law for the shuffled series
ks_in, ps_in, ks_valid_in, gamma_in, C_in = compute_degree_distribution(shuffled_series, k_min=interval[0][0], k_max=interval[0][1])

# Plot degree distribution and fitted power-law for the shuffled series in the inset
ax_inset.scatter(ks_in, ps_in, s=4, facecolors='none', edgecolors=colors[0], linewidth=0.4, alpha=0.5)

# Plot the fitted power-law curve for the shuffled series in the inset
ks_fit_in = np.logspace(np.log10(ks_valid_in.min()), np.log10(ks_valid_in.max()), 200)

# Plot the fitted power-law curve for the shuffled series in the inset
ax_inset.plot(ks_fit_in, power_law(ks_fit_in, gamma_in, C_in), '--', color=colors[1], lw=1.5)

# Set log-log scale, labels, ticks, and annotation for the inset plot
ax_inset.set_xscale("log"); ax_inset.set_yscale("log")
ax_inset.tick_params(axis='both', which='both', top=True, right=True, labelsize=text_size-4, direction='in')
ax_inset.set_xlabel(r'$k$',fontsize=text_size-4)
ax_inset.set_ylabel(r'$p(k)$',fontsize=text_size-4)
ax_inset.text(0.20, 0.4, rf"$\gamma = {gamma_in:.2f}$", transform=ax_inset.transAxes, fontsize=text_size-4, color=colors[1])

# ==== Plot c ====
# Compute degree distribution and fit power-law for series_c
ks_c, ps_c, ks_valid_c, gamma_c, C_c = compute_degree_distribution(series_c, k_min=interval[0][0], k_max=interval[0][1])

# Plot degree distribution and fitted power-law for series_c
ax_c.scatter(ks_c, ps_c, s=7.2, facecolors='none', edgecolors=colors[0], linewidth=0.6, alpha=0.5)

# Plot the fitted power-law curve for series_c
ks_fit_c = np.logspace(np.log10(ks_valid_c.min()), np.log10(ks_valid_c.max()), 200)

# Plot the fitted power-law curve for series_c
ax_c.plot(ks_fit_c, power_law(ks_fit_c, gamma_c, C_c), '--', color=colors[1], lw=2)

# Set log-log scale, labels, ticks, and annotation for plot c
ax_c.set_xscale("log"); ax_c.set_yscale("log")
ax_c.set_xlabel(r"$k$") 
ax_c.set_ylabel(r"$p(k)$",  labelpad=12)
ax_c.text(0.4, 0.75, rf"$\gamma = {gamma_c:.2f}$", transform=ax_c.transAxes, fontsize=text_size, color=colors[1])
ax_c.tick_params(axis='both', which='both', direction='in', top=True, right=True)

# Add panel labels
ax_a.text(-0.10, 1.1, "A", transform=ax_a.transAxes, fontsize=text_size+2, fontweight='bold')
ax_b.text(-0.10, 1.1, "B", transform=ax_b.transAxes, fontsize=text_size+2, fontweight='bold')
ax_c.text(-0.10, 1.1, "C", transform=ax_c.transAxes, fontsize=text_size+2, fontweight='bold')

print("Saving figure...")
fig.savefig('FIGURES/Figure_4.pdf')
