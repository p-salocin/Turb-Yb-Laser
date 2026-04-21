# Created Date: Thursday, April 16th 2026, 10:14:12 am
# Author: Nicolas Pessoa

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.

# Import necessary libraries
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from low_pass_filter import lp_filter


base_path = Path('DATA/PCA_TIME_SERIES/MULTI_PC_COMP')

files = {
    'QML_1': 'QML_PC_1_TIME_SERIES_286_0.npy',
    'QML_2': 'QML_PC_2_TIME_SERIES_286_0.npy',
    'QML_3': 'QML_PC_3_TIME_SERIES_286_0.npy'
}

panel_labels = ['A', 'B', 'C']
y_labels = [r'$x_{PCA_{(1)}}(t)$', r'$x_{PCA_{(1+2)}}(t)$', r'$x_{PCA_{(1+2+3)}}(t)$']

cmap = plt.cm.winter
colors_list = [cmap(0), cmap(0.3), cmap(0.6)][::-1]

series, filtered, residue = {}, {}, {}
for key, fname in files.items():
    data = np.load(base_path / fname)
    f = lp_filter(data)
    r = data.flatten() - f

    series[key] = data.flatten()
    filtered[key] = f
    residue[key] = r

text_size = 10 # Adequade size text
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': text_size,
    'axes.linewidth': 1.05,
})

fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.2), 
                         sharex=True, 
                         constrained_layout=True)

for idx, key in enumerate(['QML_1', 'QML_2', 'QML_3']):
    ax = axes[idx]
    t = np.arange(len(series[key])) / 1e4
    ax.plot(t, residue[key], label='Original', color=colors_list[idx], linewidth=0.1, alpha=0.7)
    ax.set_ylabel(y_labels[idx])
    if idx == 2:
        ax.set_xlabel(r'$t~(\times 10^4)$')
    ax.set_xlim(min(t), max(t))
    ax.set_ylim(-4500, 4500)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.text(-0.10, 1.1, panel_labels[idx], transform=ax.transAxes, fontsize=text_size + 2, fontweight='bold')

plt.savefig('FIGURES/SM_Figure_3.pdf', dpi=300)


