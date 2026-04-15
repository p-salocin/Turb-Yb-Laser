# Created Date: Thursday, January 15th 2026, 11:17:25 am 
# Authors: Iván R. R. Gonzáles and Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose, 
# provided that proper credit is given to the original authors.

# Import necessary libraries
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Import custom modules
from low_pass_filter import lp_filter
from plot_3d import plottable_3d_info

# Define styling plot functions
def style_axis(ax: plt.Axes, time, panel_label, ylabel=None, xlabel=None):
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.tick_params(axis='both', direction='in', labelsize=text_size)
    ax.set_xlim(time[0], time[-1])
    ax.text(-0.10, 1.1, panel_label, transform=ax.transAxes, fontsize=text_size + 2, fontweight='bold')

def style_3d_axis(ax, idx, panel_label):
    ax.set_xlabel(r't $(\times 10^4)$')
    ax.set_ylabel('Wavelength (nm)')
    if idx == 0:
        ax.set_zlabel('Intensity (a.u)', rotation=-90)
    ax.view_init(34, 35)
    ax.invert_xaxis()
    ax.text2D(-0.10, 1.05, panel_label, transform=ax.transAxes, fontsize=text_size + 2, fontweight='bold')

# ======================
# Paths and file names
# ======================
base_path = Path('DATA/PCA_TIME_SERIES')
data3d_path = Path('DATA/LASER_SPECTRA')

files = {
    'CW147': 'CW_PC_TIME_SERIES_147_0.npy',
    'QML286': 'QML_PC_TIME_SERIES_286_0.npy',
    'SML690': 'SML_PC_TIME_SERIES_690_0.npy'
}
data3d_files = {
    'CW147': 'b_data_147_0.npy',
    'QML286': 'b_data_286_0.npy',
    'SML690': 'b_data_690_0.npy'
}

# Select spectral ranges for 3D plots
min_values = {'CW147': 100, 'QML286': 150, 'SML690': 30}
max_values = {'CW147': 250, 'QML286': 210, 'SML690': 320}

# Number of columns to use for 3D plotting
COLUMN_COUNT = 150000 # All data

# Skip every 200th point for plotting to reduce load
skip = 200  

# ===============================================
# Load PCA time series and apply low-pass filter
# ===============================================
series, filtered, residue = {}, {}, {}
for key, fname in files.items():
    data = np.load(base_path / fname)
    f = lp_filter(data)
    r = data.flatten() - f

    series[key] = data.flatten()
    filtered[key] = f
    residue[key] = r

# =======================
# Load 3D surface data
# =======================
surface_data = {}
for key, fname in data3d_files.items():
    data = np.load(data3d_path / fname)
    df = pd.DataFrame(data)
    X, Y, Z = plottable_3d_info(df, min_values[key], max_values[key], COLUMN_COUNT)
    surface_data[key] = (X, Y, Z)

# =======================
# Style configuration
# =======================
text_size = 8 # Adequade size text
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': text_size,
    'axes.linewidth': 1.05,
})
colors = ["#E63946", '#0000FF']

# =======================
# Create figure
# =======================

scaling_factor = 1 # If you want to adjust the overall size of the figure, change this factor. Default is 1 (original size).

fig = plt.figure(figsize=(scaling_factor * 7.2, scaling_factor * 4.8))

# Create a 3x3 grid with specified height ratios
gs = fig.add_gridspec(3, 3, height_ratios=[2, 0.6, 0.6])

# =======================
# Axis placement
# =======================

# 3D panels
ax_a = fig.add_subplot(gs[0, 0], projection='3d')
ax_b = fig.add_subplot(gs[0, 1], projection='3d')
ax_c = fig.add_subplot(gs[0, 2], projection='3d')

# Signal panels
ax_d = fig.add_subplot(gs[1, 0])
ax_e = fig.add_subplot(gs[1, 1])
ax_f = fig.add_subplot(gs[1, 2])

# Residual panels
ax_g = fig.add_subplot(gs[2, 0], sharex=ax_d)
ax_h = fig.add_subplot(gs[2, 1], sharex=ax_e)
ax_i = fig.add_subplot(gs[2, 2], sharex=ax_f)

# Group axes for easier access in loops
axes_3d = [ax_a, ax_b, ax_c]
axes_signal = [ax_d, ax_e, ax_f]
axes_residual = [ax_g, ax_h, ax_i]

# Order of plotting
orden = ['CW147', 'QML286', 'SML690']
labels = iter(['A','D','G',
               'B','E','H',
               'C','F','I'])

# =======================
# Plotting loop
# =======================
for idx, key in enumerate(orden):

    # ----- 3D surface -----
    X, Y, Z = surface_data[key]
    ax3d = axes_3d[idx]

    ax3d.plot_surface(
        X[:, ::skip]/1e4, # Scale down x-axis by 10000
        Y[:, ::skip],
        Z[:, ::skip]/1e3, # Scale down z-axis by 1000
        cmap='gnuplot2', rcount=500, linewidth=0, antialiased=False)
        

    style_3d_axis(ax3d, idx, next(labels))

    # ----- Time series -----
    x = series[key]
    f = filtered[key]
    r = residue[key]
    t = np.arange(len(x)) / 1e4

    # Signal
    ax_sig = axes_signal[idx]
    ax_sig.plot(t, x, color=colors[1], linewidth=0.5, alpha=0.5)
    ax_sig.plot(t, f, color=colors[0], linewidth=0.85)


    style_axis(
        ax_sig,
        time=t,
        panel_label=next(labels),
        ylabel=r'$I_{PCA}(t)$' if idx == 0 else None,
        xlabel=r'$t~(\times 10^4)$'
    )

    # Residual
    ax_res = axes_residual[idx]
    ax_res.plot(t, r, color=colors[1], linewidth=0.5, alpha=0.5)

    style_axis(ax_res, time=t, panel_label=next(labels), ylabel=r'$x(t)$' if idx == 0 else None, xlabel=r'$t~(\times 10^4)$')

# Adjust layout
fig.subplots_adjust(left=0.085, 
                    bottom = 0.104, 
                    right=0.923, 
                    top = 0.968,
                    wspace=0.317,
                    hspace=0.65)

print("Saving figure...")
fig.savefig('FIGURES/Figure_1.pdf')
