#Created Date: Friday, January 9th 2026, 7:02:18 pm
#Author: Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.


# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# import pylustrator
# pylustrator.start()

# Importing custom functions from other files
from Metagraph_plot import plot_metagraph_with_umap_clusters


# Load precomputed data
data = np.load('DATA/GRAFO_DATA/plot_series_data.npz', allow_pickle=True)
files_a = [rf'DATA/GRAFO_DATA/QML_286_plot/K_subplot2_line{i}.csv' for i in range(1, 6)]
files_b = [rf'DATA/GRAFO_DATA/QML_286_plot/K_subplot1_line{i}.csv' for i in range(1, 6)]

# Extract and set variables
series = data["series"]
cluster_time_indices_2d = data["cluster_time_indices_2d"]
labels_2d = data["labels_2d"]
K = int(data["K"])
window_size = 80
dfs_a = [pd.read_csv(f, header=None) for f in files_a]
dfs_b = [pd.read_csv(f, header=None) for f in files_b]

# Plot settings

# Select font and style parameters
text_size = 8
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': text_size,
    'axes.linewidth': 1.05
})

# Define cluster and line colors
cmap = plt.cm.inferno
cluster_colors = ['#523030', '#FF0000', '#FFFF00']
colors = ['#00FF00', '#FFFF00', '#FF0000', '#523030', '#0000FF']
line_styles = ['solid', 'dashed', 'dashed', 'dashed']

# Create the figure and axes
fig = plt.figure(figsize=(3.5, 3.8), 
                 constrained_layout=False)

# Define grid specification and span of the subplots
gs = fig.add_gridspec(2,2, width_ratios=[1, 1], height_ratios=[1, 1.2])

ax0 = fig.add_subplot(gs[0, 0]) 
ax1 = fig.add_subplot(gs[0, 0])        # Metagraph
ax2 = fig.add_subplot(gs[0, 1])        # Time series and pie chart     
ax3 = fig.add_subplot(gs[1, 0])        # P(x)
ax4 = fig.add_subplot(gs[1, 1])        # f(ε)

ax0.axis('off')
ax2.axis('off') #Turn off

# Subdivision of ax2 into two insets: one for the time series and another for the pie chart
sub_gs = gs[0, 1].subgridspec(2, 1,  height_ratios=[1, 2])
ax2_top = fig.add_subplot(sub_gs[0, 0])  # Pie chart
ax2_bottom = fig.add_subplot(sub_gs[1, 0])  # Time series

# Plot a) Metagraph with UMAP clusters
res_meta = plot_metagraph_with_umap_clusters(
    series=series,
    labels_windows=labels_2d,
    window_size=window_size,
    colors=cluster_colors,
    nivel="auto",
    seed=42,
    scale=4.8,
    ax=ax1)

# Plot b) Time series with cluster coloring

# Pie chart inset for cluster proportions
M_k = np.bincount(labels_2d, minlength=K)
labels_pie = [f'Cluster {i+1}' for i in range(K)]
ax2_top.pie(M_k, labels=labels_pie, startangle=90, counterclock=False, colors=cluster_colors, labeldistance=1.2, textprops={'fontsize' : text_size-2})
ax2_top.set_aspect('equal', 'box')
ax2_top.axis('off') #Turn of frame

for k in range(K):
    idx = cluster_time_indices_2d[k]
    if idx.size:
        ax2_bottom.scatter(idx/10000, series[idx], s=0.05, color=cluster_colors[k], alpha=0.90, zorder=0-k)


ax2_bottom.set_xlabel(r'$t~(\times 10^4~\text{a.u.})$')
ax2_bottom.set_ylabel(r'$\varepsilon(t)$')
ax2_bottom.tick_params(axis='both', which='both', direction='in')
ax2_bottom.set_xlim(0,15)
ax2_bottom.set_ylim(-1, 10)


# Plot c): P(x)
for i, df in enumerate(dfs_b):
    if i < 4:
        ax3.plot(df.iloc[:, 0], df.iloc[:, 1],
                     lw=1.2, color=colors[i], ls=line_styles[i], zorder=i-4 if i != 0 else 0)
    else:
        ax3.scatter(df.iloc[:, 0], df.iloc[:, 1],
                        linewidth=0.9, marker='o', facecolors='none',
                        color=colors[i], s=5, zorder=-6)

ax3.set_ylabel(r"$P(x)$")
ax3.set_xlabel(r"$x$")
ax3.set_yscale('log')
ax3.set_xlim(-10,10)
ax3.set_ylim(1e-5,3e0)
ax3.tick_params(axis='both', which='both', top=True, right=True, direction='in')


# Plot d) f(ε) 
for i, df in enumerate(dfs_a):

    if i < 4:
        ax4.plot(df.iloc[:, 0], df.iloc[:, 1],
                     lw=1.2, color=colors[i], ls=line_styles[i], zorder=i-4 if i != 0 else 0)
    else:
        ax4.scatter(df.iloc[:, 0], df.iloc[:, 1],
                        linewidth=0.9, marker='o', facecolors='none',
                        color=colors[i], s=5, zorder=-6)

ax4.set_ylabel(r"$f(\epsilon)$")
ax4.set_xlabel(r"$\epsilon$")
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlim(1e-2,2.5e1)
ax4.set_ylim(8e-5,3e1)
ax4.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=text_size)


ax0.text(-0.12, 1.1, "A", transform=ax0.transAxes, ha='right', va='bottom', fontsize=text_size+2, fontweight='bold')
ax2.text(-0.12, 1.1, "B", transform=ax2.transAxes, ha='right', va='bottom', fontsize=text_size+2, fontweight='bold')
ax3.text(-0.12, 1.1, "C", transform=ax3.transAxes, ha='right', va='bottom', fontsize=text_size+2, fontweight='bold')
ax4.text(-0.12, 1.1, "D", transform=ax4.transAxes, ha='right', va='bottom', fontsize=text_size+2, fontweight='bold')

fig.subplots_adjust(left=0.14, 
                    bottom = 0.11, 
                    right=0.97, 
                    top = 0.92,
                    wspace=0.5,
                    hspace=0.45)

plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
plt.figure(1).axes[1].set(position=[0.06377, 0.456, 0.4845, 0.5292])

print("Saving figure...")
fig.savefig('FIGURES/Figure_3.pdf')