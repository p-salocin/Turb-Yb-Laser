# Created Date: Wednesday, April 1st 2026, 5:48:32 pm
# Author: Nicolas Pessoa

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.

# Importing necessary libraries
import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl

# Path settings
data_path = 'DATA/PCA_RESULTS'
total_list = [147, 286, 690]
label_names = ['CW', 'QML', 'SML']
markers = ['s', 'o', '^']

# Color and plot settings
cmap = plt.cm.viridis
colors_list = [cmap(0.15), cmap(0.50), cmap(0.85)]

text_size = 10
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': text_size,
    'axes.linewidth': 1.05,})

# Create the figure and axes
plt.figure(figsize=(6,4), constrained_layout=True)

for k in total_list:

    file = os.path.join(data_path, f'cumvar_{k}.npy')

    if not os.path.exists(file):
        print(f'Missing file for dataset {k}')
        continue

    cumulative_variance = np.load(file)

    plt.plot(range(1,6), cumulative_variance[:5]*100, marker=markers[total_list.index(k)], lw=2, 
             label=f'{label_names[total_list.index(k)]}', 
             color=colors_list[total_list.index(k)])


# Final plot adjustments
plt.xlabel('Principal Component (PC)')
plt.ylabel('Cumulative Explained Variance (%)')
plt.xticks(range(1,6))
plt.ylim(40,105)
plt.xlim(0.9,5.1)
plt.legend(loc='center right')
plt.tick_params(axis='both', which='both', direction='in')
plt.savefig('FIGURES/SM_Figure_2.pdf')