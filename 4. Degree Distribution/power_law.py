# Created Date: Wednesday, April 15th 2026, 12:35:58 pm
# Author: Iván R. R. Gonzáles and Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.

# Importing necessary libraries
import numpy as np

# === Power law function ===
def power_law(x, gamma, C):
    """
    Computes a power-law distribution.
        
    Parameters:
    -----------
    x : float or array
        The input value(s) at which to evaluate the power law
    gamma : float
        The power-law exponent (typically > 0). 
        Larger gamma values lead to steeper decay.
    C : float
        The normalization constant or coefficient that scales the output
    
    Returns:
    --------
    float or array
        The power-law value(s) computed as C * x^(-gamma)
    """
    return C * np.power(x, -gamma)