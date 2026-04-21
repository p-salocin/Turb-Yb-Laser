# Created Date: Thursday, January 15th 2026, 11:22:47 am
# Authors: Iván R. R. Gonzáles and Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original authors.


# Import necessary libraries
import numpy as np

def lp_filter(x: np.ndarray, win: int = 201):
    """
    Apply a low-pass filter using a moving average with reflect padding.
    
    Parameters
    ----------
    x : array-like
        Input signal to be filtered. Will be converted to a 1D numpy array.
    win : int, optional
        Window size for the moving average filter. Default is 201.

    Returns
    -------
    y : ndarray
        Filtered signal as a 1D numpy array with the same length as input.
        
    """
    # Convert input to 1D numpy array
    x = np.asarray(x).flatten()
    
    # Ensure window size is odd and not larger than ~2/3 of signal length
    win = int(max(3, min(win, len(x) // 3 * 2 + 1)))
    if win % 2 == 0:
        win += 1
    
    # Create kernel with equal weights (moving average)
    ker = np.ones(win) / win
    
    # Calculate padding size for symmetric borders
    pad = win // 2
    
    # Add reflect padding to handle edges smoothly
    xpad = np.pad(x, (pad, pad), mode='reflect')
    
    # Apply convolution and return valid region
    y = np.convolve(xpad, ker, mode='valid')
    
    return y


