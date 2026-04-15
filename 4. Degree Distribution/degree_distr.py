# Created Date: Wednesday, April 15th 2026, 12:36:18 pm
# Author: Iván R. R. Gonzáles and Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.

# Importing necessary libraries
import numpy as np
from ts2vg import NaturalVG
from scipy.optimize import curve_fit

# Importing custom functions
from power_law import power_law


def compute_degree_distribution(temporal_series, k_min, k_max):
    '''
    Computes the degree distribution of a visibility graph constructed from a temporal series,
    and fits a power-law function to the valid portion of the distribution.
    
    Parameters:
    -----------
    temporal_series : array-like
        The input temporal series from which to construct the visibility graph.
    k_min : int, optional
        The minimum degree (k) to consider for fitting the power-law.
    k_max : int, optional
        The maximum degree (k) to consider for fitting the power-law.
    
    Returns:
    --------
    ks : array
        The unique degree values in the visibility graph.
    ps : array
        The corresponding probabilities of each degree value.
    ks_valid : array
        The subset of degree values that are valid for fitting (where ps > 0 and within [k_min, k_max]).
    gamma : float
        The estimated power-law exponent from the fit.
    '''

    # Compute the cumulative sum of the temporal series
    NS = np.cumsum(temporal_series)

    # Build the natural visibility graph and compute the degree distribution
    g = NaturalVG().build(NS, only_degrees=True)

    # Extract the degree values (ks) and their corresponding probabilities (ps)
    ks, ps = g.degree_distribution

    # Filter for the specified range of degrees and where probabilities are positive
    valid = (ps > 0) & (ks >= k_min) & (ks <= k_max)
    ks_valid, ps_valid = ks[valid], ps[valid]

    # Fit the power-law function to the valid portion of the degree distribution
    gamma, C = curve_fit(power_law, ks_valid, ps_valid)[0]

    return ks, ps, ks_valid, gamma, C
