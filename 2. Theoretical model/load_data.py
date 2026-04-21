# Created Date: Tuesday, April 14th 2026, 3:58:51 pm
# Author: Iván R. R. Gonzáles and Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.

# Import necessary libraries
import scipy.io
import numpy as np

def load_data(file_path, type_archive):
    """
    Load data from a file based on the specified archive type.

    Parameters:
    file_path (str): The path to the file to load data from.
    type_archive (str): The type of the archive ('mat' for MATLAB files, otherwise NumPy files).

    Returns:
    The loaded data (dict for 'mat' files, array for others).
    """
    if type_archive == 'mat':
        # Load MATLAB .mat file using scipy
        mat_contents = scipy.io.loadmat(file_path)
        # Extract the specific variable 'QML286' from the loaded data
        load_data = mat_contents['QML286']
    else:
        # Load NumPy .npy or .npz file
        load_data = np.load(file_path)

    print("data successfully loaded")

    return load_data