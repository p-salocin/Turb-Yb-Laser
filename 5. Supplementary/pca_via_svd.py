
# Filename: c:\Users\deskl\OneDrive\Área de Trabalho\03 BENCH\PCA Analysis\main.py
# Path: c:\Users\deskl\OneDrive\Área de Trabalho\03 BENCH\PCA Analysis
# Created Date: Wednesday, September 4th 2024, 12:31:54 pm
# Author: Lab. Metrologia Óptica UFPE

# Copyright (c) 2024 Your Company

# compute_pca.py
import os
import numpy as np

input_path = 'DATA/LASER_SPECTRA'
output_path = 'DATA/PCA_RESULTS'

os.makedirs(output_path, exist_ok=True)

total_list = [147, 286, 690]
l = 0

for k in total_list:

    filename = f'b_data_{k}_{l}.npy'
    filepath = os.path.join(input_path, filename)

    if not os.path.exists(filepath):
        print(f'Error: File "{filepath}" does not exist')
        continue

    print(f'Processing dataset {k}...')

    with open(filepath, 'rb') as f:
        data = np.transpose(np.load(f))[1:]

    centered_data = data - np.mean(data, axis=0)

    U, S, VT = np.linalg.svd(centered_data, full_matrices=0)

    explained_variance = (S**2) / np.sum(S**2)
    cumulative_variance = np.cumsum(explained_variance)

    # Save results
    save_file = os.path.join(output_path, f'cumvar_{k}.npy')
    np.save(save_file, cumulative_variance)

print("All PCA cumulative variance results saved.")