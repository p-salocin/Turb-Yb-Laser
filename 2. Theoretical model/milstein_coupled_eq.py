# Created Date: Tuesday, April 14th 2026, 12:24:44 pm
# Author: Iván R. R. Gonzáles

# This code is free to use, modify, and distribute for any purpose,
# provided that proper credit is given to the original author.


# Import necessary libraries
import numpy as np
import math


def milstein_coupled_with_stochastic_variance(
    N=50_000, dt=1e-3, seed=42,
    I10=1.0, I20=0.5, nu10=0.05, nu20=0.05,
    g11_2=1.0, g11_4=-0.7, g12_4=-0.5,
    g22_2=1.0, g22_4=-0.5,
    nu01=0.05, nu02=0.05,
    gamma1=2.0, gamma2=2.0,
    kappa1=0.4, kappa2=0.4,
    clamp_nonneg=True):
    
    """
    Solves coupled stochastic differential equations using Milstein scheme with state-dependent noise.
    
    Models two coupled intensity variables (I1, I2) with time-varying stochastic variances (nu1, nu2).
    Uses the Milstein method (strong order 1) for accurate integration of multiplicative noise.
    
    Parameters:
    -----------
    N : int
        Number of time steps
    dt : float
        Time step size
    seed : int
        Random seed for reproducibility
    I10, I20 : float
        Initial conditions for intensities I1, I2
    nu10, nu20 : float
        Initial conditions for stochastic variances
    g11_2, g11_4, g12_4, g22_2, g22_4 : float
        Cubic and quintic coupling coefficients in drift terms
    nu01, nu02 : float
        Mean reversion levels for variances
    gamma1, gamma2 : float
        Mean reversion rates (speed of variance relaxation)
    kappa1, kappa2 : float
        Volatility of volatility parameters (noise strength on variances)
    clamp_nonneg : bool
        If True, clamps intensities to [0, ∞) and variances to [1e-14, ∞) for stability
    
    Returns:
    --------
    I1, I2 : np.ndarray
        Time series of coupled intensities
    nu1, nu2 : np.ndarray
        Time series of stochastic variances
    params : dict
        Dictionary of model parameters
    """
    
    # Initialize random number generator
    rng = np.random.default_rng(seed)
    
    # Allocate arrays for time series
    I1 = np.empty(N, dtype=float); I2 = np.empty(N, dtype=float)
    nu1 = np.empty(N, dtype=float); nu2 = np.empty(N, dtype=float)
    
    # Set initial conditions
    I1[0], I2[0] = float(I10), float(I20)
    nu1[0], nu2[0] = float(nu10), float(nu20)
    
    # Precompute sqrt(dt) for Wiener increments
    sqdt = math.sqrt(dt)

    # Main integration loop
    for n in range(N-1):
        # Get current state variables
        x1, x2 = I1[n], I2[n]
        v1, v2 = nu1[n], nu2[n]
        
        # Clamp to ensure numerical stability (prevent negative variances and intensities)
        if clamp_nonneg:
            x1 = max(x1, 0.0); x2 = max(x2, 0.0)
            v1 = max(v1, 1e-14); v2 = max(v2, 1e-14)

        # Compute drift terms (deterministic nonlinear coupling)
        # f1, f2 represent the cubic/quintic growth dynamics with cross-coupling
        f1 = x1 * (g11_2 + g11_4 * x1 + g12_4 * x2)
        f2 = x2 * (g22_2 + g22_4 * x2 + g12_4 * x1)

        # Generate independent Wiener increments for intensities
        dW1 = rng.normal(0.0, sqdt)
        dW2 = rng.normal(0.0, sqdt)
        
        # Compute diffusion coefficients (sqrt of variances)
        gI1 = math.sqrt(max(v1, 0.0))
        gI2 = math.sqrt(max(v2, 0.0))
        
        # Milstein correction terms: second-order terms from Itô expansion
        # Accounts for multiplicative noise: d(x*sqrt(v)) requires Milstein scheme
        m_I1  = 0.5 * v1 * x1 * (dW1**2 - dt)
        m_I2  = 0.5 * v2 * x2 * (dW2**2 - dt)

        # Milstein update for intensities: Euler drift + multiplicative noise + Milstein correction
        I1_next = x1 + f1 * dt + x1 * gI1 * dW1 + m_I1
        I2_next = x2 + f2 * dt + x2 * gI2 * dW2 + m_I2

        # Generate independent Wiener increments for variances
        dW3 = rng.normal(0.0, sqdt)
        dW4 = rng.normal(0.0, sqdt)

        # Milstein update for variances: mean-reverting process with multiplicative noise
        # Favors relaxation toward nu0i with rate gamma, plus stochastic perturbations
        nu1_next = v1 + (-gamma1*(v1 - nu01)) * dt + kappa1 * v1 * dW3 + 0.5 * (kappa1**2) * v1 * (dW3**2 - dt)
        nu2_next = v2 + (-gamma2*(v2 - nu02)) * dt + kappa2 * v2 * dW4 + 0.5 * (kappa2**2) * v2 * (dW4**2 - dt)

        # Clamp updated values to ensure physical validity and numerical stability
        if clamp_nonneg:
            I1_next = max(I1_next, 0.0)
            I2_next = max(I2_next, 0.0)
            nu1_next = max(nu1_next, 1e-14)
            nu2_next = max(nu2_next, 1e-14)

        # Store updated values in time series arrays
        I1[n+1], I2[n+1] = I1_next, I2_next
        nu1[n+1], nu2[n+1] = nu1_next, nu2_next

    # Compile parameters for reference
    params = dict(
        g11_2=g11_2, g11_4=g11_4, g12_4=g12_4, g22_2=g22_2, g22_4=g22_4,
        nu01=nu01, nu02=nu02, gamma1=gamma1, gamma2=gamma2, kappa1=kappa1, kappa2=kappa2
    )
    
    return I1, I2, nu1, nu2, params