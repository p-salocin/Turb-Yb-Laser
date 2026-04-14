# Created Date: Tuesday, April 14th 2026, 12:20:15 pm
# Author: Iván R. R. Gonzáles

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.

# Import necessary libraries
import numpy as np


def histogram_sim(rho1, ymin, ymax):
    """
    Compute a symmetric normalized histogram with optional negative data reflection.
    
    Parameters:
    -----------
    rho1 : array_like
        Input data array for histogram
    ymin : float
        Minimum value for histogram range
    ymax : float
        Maximum value for histogram range
    
    Returns:
    --------
    H : ndarray
        2D array where column 0 is bin centers and column 1 is normalized probabilities
    """
    # Calculate bin width from range divided into 80 segments
    dybin = (ymax - ymin) / 80
    
    # Create bin edges centered between ymin and ymax with 70 bins
    ybin = np.linspace(ymin + dybin / 2, ymax - dybin / 2, 70)
    
    # Histogram of positive input data
    qq1, bb1 = np.histogram(rho1, bins=ybin)
    
    # Calculate area (integral) of first histogram
    Ar1 = sum(dybin * qq1)
    
    # Normalize histogram to probability density (handle zero area case)
    pp1 = qq1 / Ar1 if Ar1 > 0 else np.zeros_like(qq1)
    
    # Histogram of negated data (reflection about zero)
    qq2, bb2 = np.histogram(-rho1, bins=ybin)
    
    # Calculate area of second histogram
    Ar2 = sum(dybin * qq2)
    
    # Normalize second histogram to probability density
    pp2 = qq2 / Ar2 if Ar2 > 0 else np.zeros_like(qq2)
    
    # Average the two normalized histograms (creates symmetric distribution)
    pp = 0.5 * (pp2 + pp1)
    
    # Get number of bins
    Th = len(pp)
    
    # Create output array with bin centers and probabilities
    H = np.zeros((Th, 2))
    
    # Column 0: bin center positions
    H[:, 0] = (bb1[:-1] + bb1[1:]) / 2
    
    # Column 1: averaged probability density
    H[:, 1] = pp
    
    return H