# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 19:05:43 2025

@author: Guillaume
"""

import numpy as np

def hjorth_parameters(signal, sampling_rate=1.0):
    """
    Compute Hjorth Activity, Mobility, and Complexity for a 1D signal.

    Parameters
    ----------
    signal : array_like
        1D time series (e.g., EEG channel).
    sampling_rate : float, optional
        Sampling frequency in Hz. Only affects scaling of derivatives;
        if you care only about relative values, you can leave it at 1.0.

    Returns
    -------
    activity : float
        Hjorth Activity (variance of the signal).
    mobility : float
        Hjorth Mobility.
    complexity : float
        Hjorth Complexity.
    """
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]  # remove NaN/Inf if any

    if x.size < 3:
        raise ValueError("Signal too short to compute Hjorth parameters.")

    # Activity: variance of the signal
    activity = np.var(x, ddof=1)

    # First derivative (discrete)
    # Using central differences via np.diff is fine; dt = 1/sampling_rate
    dt = 1.0 / sampling_rate
    dx = np.diff(x) / dt

    # Second derivative
    d2x = np.diff(dx) / dt

    var_dx = np.var(dx, ddof=1)
    var_d2x = np.var(d2x, ddof=1)

    # Mobility of signal and of its first derivative
    mobility = np.sqrt(var_dx / activity) if activity > 0 else 0.0
    mobility_derivative = np.sqrt(var_d2x / var_dx) if var_dx > 0 else 0.0

    # Complexity = mobility(derivative) / mobility(signal)
    complexity = mobility_derivative / mobility if mobility > 0 else 0.0

    return activity, mobility, complexity
