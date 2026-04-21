# Created Date: Tuesday, April 14th 2026, 11:43:09 am
# Author: Iván R. R. Gonzáles and Nicolas P. Alves

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.

# Import necessary libraries
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d


def butterworth_filter(data, fs=100, fc=1, window_size=2000):
    """
    Low-pass Butterworth filter with baseline removal.
    
    Applies a 4th-order low-pass Butterworth filter to the input data with symmetric
    extension to minimize edge effects. The filtered signal is then compared against
    a smoothed baseline to extract high-frequency fluctuations.
    
    Parameters:
    -----------
    data : array_like
        Input signal to be filtered
    fs : float, optional
        Sampling frequency in Hz (default: 100)
    fc : float, optional
        Cutoff frequency in Hz (default: 1)
    window_size : int, optional
        Window size for uniform filtering used to compute baseline (default: 2000)
    
    Returns:
    --------
    DF : ndarray
        Detrended signal (data minus baseline)
    D : ndarray
        Baseline (smoothed filtered signal)
    """
    # Generate time array
    t = np.arange(len(data))
    
    # Design 4th-order low-pass Butterworth filter
    b, a = butter(4, fc / (fs / 2), 'low')
    
    # Extend data symmetrically to reduce edge effects
    n_extend = max(1, len(data) // 4)
    data_extended = np.concatenate([
        data[0] + (data[0] - data[1:n_extend+1])[::-1],
        data,
        data[-1] + (data[-1] - data[-n_extend-1:-1])[::-1]
    ])
    
    # Apply forward-backward filter for phase preservation
    data_filtered = filtfilt(b, a, data_extended)
    
    # Extract the original data portion from extended array
    start_idx = len(data_extended) - len(data) - n_extend
    end_idx = start_idx + len(data)
    data_filtered = data_filtered[start_idx:end_idx]
    
    # Compute baseline using uniform (moving average) filter
    D = uniform_filter1d(data_filtered, size=window_size, mode='reflect')
    
    # Calculate detrended signal (fluctuations around baseline)
    DF = data - D
    
    return DF, D