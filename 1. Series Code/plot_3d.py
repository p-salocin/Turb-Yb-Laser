# Created Date: Monday, March 23rd 2026, 3:22:31 pm
# Authors: Iván R. R. Gonzáles and Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original authors.

# Import necessary libraries
import numpy as np
import pandas as pd

def plottable_3d_info(df: pd.DataFrame, min_value: int, max_value: int, column_count: int):
    """
    Transform a Pandas DataFrame into a format compatible with Matplotlib's surface and wireframe plotting.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be plotted.
        min_value (int): Minimum index for slicing the data.
        max_value (int): Maximum index for slicing the data.
        column_count (int): Number of columns to use for plotting.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: 
            x, y, Z meshgrids.

    """
    # Create an array for the x-axis based on the number of columns
    x = np.arange(column_count)

    # Wavelengths
    y = df.iloc[min_value:max_value, 0].values

    # Intensity columns
    Z = df.iloc[min_value:max_value, 1:1 + column_count].values  

    # Ensure proper meshgrid shapes
    X, Y = np.meshgrid(x, y)

    return X, Y, Z