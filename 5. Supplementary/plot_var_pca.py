# plot_pca.py
import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl

data_path = 'DATA/PCA_RESULTS'

total_list = [147, 286, 690]

# Color settings for the plot
cmap = plt.cm.inferno
colors_list = [cmap(0.15), cmap(0.50), cmap(0.85)]
label_names = ['CW', 'QML', 'SML']

text_size = 12 # Adequade size text
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'font.size': text_size,
    'axes.linewidth': 1.05,})

plt.figure(figsize=(6,4), constrained_layout=True)

for k in total_list:

    file = os.path.join(data_path, f'cumvar_{k}.npy')

    if not os.path.exists(file):
        print(f'Missing file for dataset {k}')
        continue

    cumulative_variance = np.load(file)

    plt.plot(range(1,6), cumulative_variance[:5]*100, 'o-', lw=2, 
             label=f'{label_names[total_list.index(k)]}', 
             color=colors_list[total_list.index(k)])

plt.xlabel('Principal Component (PC)')
plt.ylabel('Cumulative Explained Variance (%)')
plt.xticks(range(1,6))
plt.ylim(40,105)
plt.xlim(0.9,5.1)
plt.legend(loc='center right')
plt.tick_params(axis='both', which='both', direction='in')
plt.savefig('FIGURES/SM_Figure_4.pdf')