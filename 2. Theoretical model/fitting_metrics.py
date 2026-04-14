# Created Date: Tuesday, April 14th 2026, 12:31:05 pm
# Author: Iván R. R. Gonzáles

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.

# Import necessary libraries
import numpy as np
from sklearn.decomposition import PCA

# Import custom modules
from filter import butterworth_filter
from histogram import histogram_sim


def calculate_fit_metrics(I1, I2, QML_data, burn_frac=0.3, ymin=-10, ymax=10):
    """
    Calculate fitting metrics comparing simulated PCA data with quasi mode-locked data (QML) data.
    
    Parameters:
    - I1, I2: Input signal arrays to be analyzed
    - QML_data: Quasi mode-locked reference data for comparison
    - burn_frac: Fraction of initial data to discard (transient behavior)
    - ymin, ymax: Histogram range bounds
    
    Returns: Dictionary with variance, error metrics, and histograms
    """
    # Calculate burn-in period: discard first burn_frac*n samples to remove transients
    n = len(I1)
    b0 = int(max(0, min(n-1, np.floor(burn_frac*n))))
    x1 = I1[b0:]; x2 = I2[b0:]
    
    # Remove non-finite values (NaN, inf) from the data
    x1 = x1[np.isfinite(x1)]; x2 = x2[np.isfinite(x2)]

    # Apply Butterworth low-pass filter to remove high-frequency noise (fc=1 Hz)
    x1_detrended, _ = butterworth_filter(x1, fs=100, fc=1, window_size=2000)
    x2_detrended, _ = butterworth_filter(x2, fs=100, fc=1, window_size=2000)

    # Stack the two detrended signals and perform PCA to extract dominant mode
    data_matrix = np.column_stack((x1_detrended, x2_detrended))
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(data_matrix).flatten()

    # Normalize the principal component by its standard deviation
    std_pc1 = np.std(pc1)
    if std_pc1 < 1e-12: std_pc1 = 1.0  # Avoid division by zero
    pc1_normalized = pc1 / std_pc1

    # Flatten and normalize QML data similarly
    QML = QML_data.flatten()
    std_qml = np.std(QML)
    if std_qml < 1e-12: std_qml = 1.0  # Avoid division by zero
    QML_normalized = QML / std_qml

    # Compute histograms for both normalized signals
    H_sim = histogram_sim(pc1_normalized, ymin, ymax)
    H_qml = histogram_sim(QML_normalized, ymin, ymax)

    # Apply floor to histogram bins to prevent log(0) issues
    H_sim[:, 1] = np.maximum(H_sim[:, 1], 1e-20)
    H_qml[:, 1] = np.maximum(H_qml[:, 1], 1e-20)

    # Create exponential weights that emphasize histogram tails (larger absolute values)
    x_abs = np.abs(H_sim[:, 0])
    weights = np.exp(x_abs / (np.max(x_abs) + 1e-9))
    weights = weights / np.sum(weights)  # Normalize to unit sum

    # Calculate log-scale weighted squared errors between histograms
    log_sim = np.log10(H_sim[:, 1])
    log_qml = np.log10(H_qml[:, 1])
    weighted_errors = weights * (log_sim - log_qml) ** 2
    mse_log_weighted = np.sum(weighted_errors)

    # Compute weighted R² coefficient of determination
    ss_res = np.sum(weighted_errors)
    ss_tot = np.sum(weights * (log_qml - np.sum(weights * log_qml)) ** 2)
    r2_log_weighted = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Return comprehensive metrics dictionary
    return {
        'explained_variance': pca.explained_variance_ratio_[0] * 100,  # PCA variance %
        'mse_log_weighted': mse_log_weighted,  # Weighted log-scale MSE
        'r2_log_weighted': r2_log_weighted,    # Weighted R² score
        'std_pca': std_pc1,                    # PCA component std dev
        'std_qml': std_qml,                    # QML data std dev
        'H_sim': H_sim,                        # Simulated histogram
        'H_qml': H_qml                         # QML reference histogram
    }