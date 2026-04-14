import re
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import ceil
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

text_size = 17

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': text_size,
})
plt.rcParams['axes.linewidth'] = 1.2

# Path to .npy files
folder_path = r'DATA/PCA_TIME_SERIES_FILT'

# Files to load (in desired order)
desired_order = [147, 690]
text_label = ['CW', 'SML']

# Use glob to get all .npy files
files = glob.glob(os.path.join(folder_path, "*.npy"))

# Regex to capture number before "_0_filtered.npy"
pattern = re.compile(r"_(\d+)_0_filtered\.npy$")

# Load selected files
data_dict = {}
for f in files:
    match = pattern.search(f)
    if match:
        file_num = int(match.group(1))
        if file_num in desired_order:
            data_dict[file_num] = np.load(f)

print(data_dict)

# Create DataFrame in the correct order
df_time_series = pd.DataFrame({num: data_dict[num] for num in desired_order})

# Compute increments
df_increment_series = df_time_series

# Standardize (z-score)
normalized_series = (df_increment_series - df_increment_series.mean()) / df_increment_series.std()

# --- Adjustable inset offsets ---
inset_x_offset = 0.015
inset_y_offset = 0.15

# --- Figure scaling ---
fig_height = 3.5 * 2.5 * len(desired_order) / 2
fig_width = 3.5 * 2.5

fig, axes = plt.subplots(len(desired_order), 1, figsize=(fig_width, fig_height), 
                         sharex=True, sharey=False)
if len(desired_order) == 1:
    axes = [axes]  # make iterable if single

colors = ["#E63946", "#164979"]
labels = ['a', 'b']

for i, col in enumerate(normalized_series.columns):
    ax = axes[i]
    x = normalized_series[col].values

    # Histogram
    bins = ceil(2 * (len(x)) ** (1 / 4))
    hist, bin_edges = np.histogram(x, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Gaussian fit
    mu, sigma = norm.fit(x)
    x_scale = np.linspace(-4.5, 4.5, 10000)
    pdf = norm.pdf(x_scale, mu, sigma)

    # Plot Gaussian fit + histogram
    ax.scatter(bin_centers, hist, linewidth=1.7, marker='o', facecolors='none',
               color=colors[1], s=55, zorder=0, label=text_label[i])
    ax.plot(x_scale, pdf, color=colors[0], linewidth=2, zorder=2, label='Gaussian fit')


    # Formatting
    ax.set_ylabel("$P(x)$", fontsize=text_size, labelpad=12)
    ax.set_yscale('log')
    ax.set_xlim(-4.3, 4.3)
    ax.set_ylim(2e-4, 1e0)
    ax.tick_params(axis='both', which='both', top=True, right=True, direction='in', labelsize=text_size)
    ax.legend(loc='lower center')

    # External subplot label (a), (b)
    ax.text(-0.12, 1.02, f"{labels[i]}", transform=ax.transAxes,
            ha='right', va='bottom', fontsize=text_size+3, fontweight='bold')

    # # --- Inset with temporal series ---
    # inset_width, inset_height = 0.5, 0.22
    # inset_ax = ax.inset_axes([
    #     0.5 - inset_width/2 + inset_x_offset,
    #     0.05 + inset_y_offset,
    #     inset_width, inset_height
    # ])
    # t = np.arange(len(normalized_series[col]))
    # inset_ax.plot(t, normalized_series[col], lw=.5, color=colors[0])  # <-- normalized here

    # # Labels inside inset
    # inset_ax.set_xlabel(r"$t~~(\times 10^4)$", fontsize=text_size-2, labelpad=2)
    # inset_ax.set_ylabel(r"$x(t)$", fontsize=text_size-2, labelpad=2)

    # inset_ax.set_xticks(np.linspace(0, len(t), 4, dtype=int))
    # inset_ax.set_xticklabels((np.linspace(0, len(t), 4) / 1e4).astype(int))
    # inset_ax.tick_params(axis='both', which='both', direction='in', labelsize=text_size-2)
    # inset_ax.set_xlim(0,150000)
    # inset_ax.set_ylim(-7,7)

# --- Shared xlabel centered at x=0 ---
fig.canvas.draw()
zero_x_display = axes[-1].transData.transform((0, 0))[0]
zero_x_axes = fig.transFigure.inverted().transform((zero_x_display, 0))[0]
fig.text(zero_x_axes + 0.045, 0.02, "$x$", ha='center', fontsize=text_size)


plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()