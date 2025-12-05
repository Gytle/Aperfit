# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 15:36:32 2025

@author: Guillaume
"""

import numpy as np
import scipy as sp
from numpy.polynomial.polynomial import polyfit
import pywt

def hurst_wavelet(
    signal,
    wavelet_name="db2",
    min_level=1,
    max_level=None,
    model="fGn",
):
    """
    Estimate the Hurst exponent H using a wavelet-variance method.

    Parameters
    ----------
    signal : array_like
        1D time series.
    wavelet_name : str, optional
        Name of the mother wavelet (e.g. "db2", "db4", "haar", "coif1", ...).
    min_level : int, optional
        Minimum decomposition level (scale) to include in the fit (>=1).
    max_level : int or None, optional
        Maximum decomposition level to include.
        If None, it uses all levels allowed by the signal length.
    model : {"fGn", "fBm"}, optional
        Assumed underlying process:
        - "fGn": fractional Gaussian noise  -> Var(W_j) ~ 2^{(2H-1)j}
        - "fBm": fractional Brownian motion -> Var(W_j) ~ 2^{(2H+1)j}
        This affects the mapping from slope -> H.

    Returns
    -------
    H : float
        Estimated Hurst exponent.
    scales : ndarray
        Scales (2^j) used in the regression.
    variances : ndarray
        Wavelet coefficient variances at each scale.
    slope : float
        Slope of log2(variance) vs log2(scale).
    """
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    N = x.size
    if N < 16:
        raise ValueError("Signal is too short for wavelet-based H estimation.")

    # 1. Discrete wavelet transform
    wavelet = pywt.Wavelet(wavelet_name)
    max_level_wt = pywt.dwt_max_level(N, wavelet.dec_len)

    if max_level is None:
        max_level = max_level_wt
    else:
        max_level = min(max_level, max_level_wt)

    if max_level < min_level:
        raise ValueError("max_level < min_level; not enough levels for estimation.")

    coeffs = pywt.wavedec(x, wavelet, level=max_level)

    # coeffs = [cA_max, cD_max, cD_{max-1}, ..., cD_1]
    # Put detail coeffs into natural order: [cD_1, cD_2, ..., cD_max]
    detail_coeffs = coeffs[1:][::-1]

    variances = []
    scales = []

    # level = 1..max_level now correctly corresponds to scale 2^level
    for level, cD in enumerate(detail_coeffs, start=1):
        if level < min_level:
            continue
        if len(cD) < 4:
            # skip very short coefficient vectors
            continue

        var = np.var(cD, ddof=1)
        variances.append(var)
        scales.append(2 ** level)

    variances = np.array(variances)
    scales = np.array(scales)

    if len(scales) < 2:
        raise ValueError("Not enough valid scales to perform regression.")

    # 2. Linear regression in log2-log2 space
    log2_scales = np.log2(scales)
    log2_vars = np.log2(variances)

    slope, intercept = np.polyfit(log2_scales, log2_vars, 1)

    # 3. Map slope -> H depending on model
    # For fGn: Var ~ scale^{2H-1} => slope = 2H - 1 => H = (slope + 1)/2
    # For fBm: Var ~ scale^{2H+1} => slope = 2H + 1 => H = (slope - 1)/2
    if model.lower() == "fgn":
        H = (slope + 1.0) / 2.0
    elif model.lower() == "fbm":
        H = (slope - 1.0) / 2.0
    else:
        raise ValueError("model must be 'fGn' or 'fBm'.")

    return H, scales, variances, slope

def DFA(signal, scales=None, order=1, overlap=False, min_scale=4, max_scale=None, n_scales=20):
    """
    Detrended Fluctuation Analysis (DFA)
    
    Parameters
    ----------
    signal : array_like
        1D time series.
    scales : array_like, optional
        Window sizes (in samples) to use. If None, they are generated
        logarithmically between min_scale and max_scale.
    order : int, optional
        Polynomial order for local detrending (1 = linear DFA1, 2 = DFA2, ...).
    overlap : bool, optional
        If True, use overlapping windows (step = scale/2).
        If False, use non-overlapping windows.
    min_scale : int, optional
        Minimum window size (if scales is None).
    max_scale : int, optional
        Maximum window size (if scales is None). If None, it defaults to N/4.
    n_scales : int, optional
        Number of scales (if scales is None).
        
    Returns
    -------
    alpha : float
        Estimated DFA scaling exponent.
    scales : ndarray
        Array of scales used (window sizes, in samples).
    fluct : ndarray
        Fluctuation function values for each scale.
    """
    x = np.asarray(signal, dtype=float)

    # Remove NaNs if any (simple: drop them)
    x = x[np.isfinite(x)]
    N = x.size
    if N < min_scale * 4:
        raise ValueError("Time series too short for DFA with given min_scale.")

    # 1. Build profile (integrated signal)
    x = x - np.mean(x)
    y = np.cumsum(x)

    # 2. Define scales if not given
    if scales is None:
        if max_scale is None:
            max_scale = N // 4
        # Log-spaced scales, rounded to integers and made unique
        scales = np.unique(
            np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales).astype(int)
        )

    fluct = []

    for s in scales:
        if s < order + 2:
            # Need at least order+2 points to fit polynomial reliably
            continue

        if overlap:
            step = s // 2
            starts = np.arange(0, N - s + 1, step)
        else:
            n_segments = N // s
            starts = np.arange(0, n_segments * s, s)

        if len(starts) == 0:
            continue

        rms_vals = []

        for start in starts:
            t = np.arange(s)
            segment = y[start+t]

            # Fit polynomial of given order
            # polyfit from numpy.polynomial.polynomial uses powers of x: c0 + c1 x + ...
            coeffs = polyfit(t, segment, order)
            trend = np.polyval(coeffs[::-1], t)  # reverse coeffs for np.polyval

            # Detrended segment
            detrended = segment - trend
            rms = np.sqrt(np.mean(detrended**2))
            rms_vals.append(rms)

        fluct.append(np.sqrt(np.mean(np.array(rms_vals) ** 2)))  # average RMS

    fluct = np.array(fluct)
    valid = fluct > 0
    scales = scales[valid]
    fluct = fluct[valid]

    # 3. Linear regression in log-log space
    log_scales = np.log(scales)
    log_fluct = np.log(fluct)

    # slope = alpha, intercept ignored
    alpha, _ = np.polyfit(log_scales, log_fluct, 1)

    return alpha, scales, fluct


def hurst_rs(x, min_chunk_size=16, max_chunk_size=None, num_scales=20):
    """
    Estimate the Hurst exponent H using rescaled range (R/S) analysis
    following the procedure on the Wikipedia Hurst exponent page.

    Parameters
    ----------
    x : array_like
        1D time series (e.g., EEG signal or PSD values).
    min_chunk_size : int
        Minimum window size n to consider.
    max_chunk_size : int or None
        Maximum window size n to consider. If None, use N//2.
    num_scales : int
        Number of scales (n values) between min_chunk_size and max_chunk_size
        on a log scale.

    Returns
    -------
    H : float
        Estimated Hurst exponent.
    details : dict
        Dictionary with intermediate values:
        - 'n': array of window sizes
        - 'RS': array of mean R/S for each n
        - 'slope': fitted slope (H)
        - 'intercept': fitted intercept (log C)
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    if max_chunk_size is None:
        max_chunk_size = N // 2

    # Choose window sizes n on a log scale
    n_values = np.floor(
        np.logspace(np.log10(min_chunk_size),
                    np.log10(max_chunk_size),
                    num=num_scales)
    ).astype(int)
    n_values = np.unique(n_values)

    RS_vals = []
    n_used  = []

    for n in n_values:
        if n < 8:
            continue

        n_chunks = N // n
        if n_chunks < 2:
            continue

        rs_chunk = []

        for i in range(n_chunks):
            seg = x[i*n : (i+1)*n]

            # 1. mean
            m = seg.mean()

            # 2. mean-adjusted series
            Y = seg - m

            # 3. cumulative deviate series
            Z = np.cumsum(Y)

            # 4. range R(n)
            R = Z.max() - Z.min()

            # 5. standard deviation S(n)
            S = np.std(seg, ddof=0)

            # 6. rescaled range R/S
            if S > 0:
                rs_chunk.append(R / S)

        if len(rs_chunk) == 0:
            continue

        RS_vals.append(np.mean(rs_chunk))
        n_used.append(n)

    n_used = np.array(n_used, dtype=float)
    RS_vals = np.array(RS_vals, dtype=float)

    # log–log regression: log(R/S) = log C + H log n
    log_n  = np.log(n_used)
    log_RS = np.log(RS_vals)

    slope, intercept = np.polyfit(log_n, log_RS, 1)
    H = slope

    details = {
        "n": n_used,
        "RS": RS_vals,
        "slope": slope,
        "intercept": intercept
    }
    return H, details

def hurst_to_slope(H):
    return 2*H