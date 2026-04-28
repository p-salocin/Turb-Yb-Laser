# Created Date: Wednesday, September 4th 2024, 12:31:54 pm
# Authors: Iván R. R. Gonzáles and Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original authors.

# Import necessary libraries
import os
import numpy as np

# Input and output paths
input_path = 'DATA/LASER_SPECTRA'
output_path = 'DATA/PCA_RESULTS'
out_path_2 = "DATA/PCA_TIME_SERIES/MULTI_PC_COMP"

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# List of dataset names to process. This correspond to CW, QML, and SML regimes, respectively.
total_list = [147, 286, 690]
files_names = ['CW_PC', 'QML_PC', 'SML_PC']

# Define the lenght of total cumulative sum of the scores in the PCA analysis
total_sum = 3

# Loop through each dataset.
for i in range(len(total_list)):

    # Construct the filename and check if it exists
    filename = f'b_data_{total_list[i]}_0.npy'
    filepath = os.path.join(input_path, filename)

    if not os.path.exists(filepath):
        print(f'Error: File "{filepath}" does not exist')
        continue

    print(f'Processing dataset {total_list[i]}...')

    # Load the data, transpose it, and remove the first column. 
    # Data is construsct in a way that the first column contains the wavelength values, 
    # which are not needed for PCA.
    with open(filepath, 'rb') as f:
        data = np.transpose(np.load(f))[1:]

    # Center the data by subtracting the mean of each column.
    centered_data = data - np.mean(data, axis=0)

    # Perform Singular Value Decomposition (SVD) on the centered data.
    U, S, VT = np.linalg.svd(centered_data, full_matrices=0)

    # List to save the scores
    scores = []

    # Save the fisrt three scores
    for j in range(total_sum):
        scores.append(S[j]*U.T[j])

    # Produce the culmulative sum for the first scores
    cumulative_scores = np.cumsum(scores, axis=0)

    for k in range(len(cumulative_scores)):
        filename = f"{files_names[i]}_{k+1}_TIME_SERIES_{total_list[i]}_0.npy"
        filepath = os.path.join(out_path_2, filename)
        #np.save(filepath, cumulative_scores[k])

    # Obtain the explained_variance
    explained_variance = (S**2) / np.sum(S**2)
    cumulative_variance = np.cumsum(explained_variance)

    # Save results
    save_file = os.path.join(output_path, f'cumvar_{total_list[i]}.npy')
    # np.save(save_file, cumulative_variance)

print("All PCA cumulative variance and scores results saved.")