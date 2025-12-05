# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 11:08:29 2025

@author: Guillaume
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import hurst

def welch_power_spec(X, fs=500, winsize=5, overlap=0.5):
    """
    Plot Welch power spectrum on a given axis.

    Parameters
    ----------
    X : 1D array
        Signal.
    fs : float
        Sampling frequency (Hz).
    winsize : float
        Window length in seconds.
    overlap : float
        Fractional window overlap (0–1).
    ax : matplotlib.axes.Axes or None
        Axis to plot into. If None, a new figure+axis is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis with the plot.
    """
    winsize_t = int(winsize * fs)
    overlapsize_t = int(winsize * fs * overlap)

    fr, pxx = sp.signal.welch(
        X,
        fs=fs,
        window="hann",
        nperseg=winsize_t,
        noverlap=overlapsize_t,
        nfft=winsize_t,
    )
    
    return fr,pxx

def fit_psd_line_with_H(X, fs=500, H=0.5,
                        fmin=1.0, fmax=None,
                        winsize=5, overlap=0.5,
                        ax=None):
    """
    Fit a 1/f^beta line to the Welch power spectrum using a given Hurst exponent.

    Assumes fractional Gaussian noise relation:
        beta = 2H - 1
        P(f) ~ 1 / f^beta

    The fit is done in log10-log10 space:
        log10(P) = intercept + slope * log10(f)
    where slope is constrained by H, and intercept is estimated.

    Parameters
    ----------
    X : 1D array
        Time series (e.g., fake EEG).
    fs : float
        Sampling frequency (Hz).
    H : float
        Hurst exponent (from hurst_rs or elsewhere).
    fmin, fmax : float or None
        Frequency range (Hz) used to fit the line.
        If fmax is None, it defaults to fs/2.
    winsize : float
        Window size (seconds) for Welch.
    overlap : float
        Overlap fraction (0–1) for Welch.
    ax : matplotlib.axes.Axes or None
        Axis to plot into. If None, a new figure and axis is created.

    Returns
    -------
    results : dict
        - 'f'           : frequencies (full PSD)
        - 'Pxx'         : PSD (linear)
        - 'f_fit'       : frequencies used for the fit
        - 'P_fit_db'    : fitted line in dB at f_fit
        - 'beta'        : spectral exponent from H (beta = 2H - 1)
        - 'slope'       : slope in log10 P vs log10 f (slope = -beta)
        - 'intercept'   : intercept in log10 space
    """
    if fmax is None:
        fmax = fs / 2.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        
    # --- 1. Welch PSD ---
    f,Pxx = welch_power_spec(X, fs=500, winsize=5, overlap=0.5)

    # --- 2. Select frequency range for fitting (avoid f=0) ---
    mask = (f > 0) & (f >= fmin) & (f <= fmax)
    f_fit = f[mask]
    P_fit = Pxx[mask]

    logf = np.log10(f_fit)
    logP = np.log10(P_fit)

    ax.plot(f,10*np.log10(Pxx))


    # H-based line (if H provided)
    if H is not None:
        
        if type(H) is list:
            
            for iH in H:
                beta_from_H = hurst.hurst_to_slope(iH)        # fractional Gaussian noise relation
    
                a_from_H = np.mean(logP + beta_from_H * logf)
                
                y_H = -beta_from_H * logf + a_from_H
                ax.plot(f_fit, y_H, '-',
                        label=f"H-based fit (H={iH:.3f}, β={beta_from_H:.2f})")
        else:
            beta_from_H = hurst.hurst_to_slope(H)        # fractional Gaussian noise relation
    
            a_from_H = np.mean(logP + beta_from_H * logf)
            
            y_H = -beta_from_H * logf + a_from_H
            ax.plot(f_fit, y_H, '-',
                    label=f"H-based fit (H={H:.3f}, β={beta_from_H:.2f})")
            

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_xlim(0, fs / 2)
    ax.legend()
    ax.set_title("PSD with 1/f fit from Hurst exponent")

    return {
        "f": f,
        "Pxx": Pxx,
        "f_fit": f_fit
    }

def fit_psd_line_loglog(X, fs=500, H=None,
                        fmin=1.0, fmax=None,
                        winsize=5, overlap=0.5,
                        ax=None):
    """
    Fit a line y = -beta * log10(f) + a in log-log space for the PSD.

    If H is provided, a 'theoretical' beta = 2H - 1 is used to build a
    constrained line (slope fixed, intercept fitted). An unconstrained
    line (free slope + intercept) is also estimated via np.polyfit.

    Parameters
    ----------
    X : 1D array
        Time series (e.g., EEG).
    fs : float
        Sampling frequency (Hz).
    H : float or None
        Hurst exponent. If provided, beta_from_H = 2H - 1 and we fit
        y = -beta_from_H * log10(f) + a_H.
    fmin, fmax : float or None
        Frequency range used for fitting. fmax defaults to fs/2.
    winsize : float
        Window size (seconds) for Welch.
    overlap : float
        Overlap fraction (0–1) for Welch.
    ax : matplotlib.axes.Axes or None
        Axis to plot into. If None, a new one is created.

    Returns
    -------
    results : dict
        - f_fit         : frequencies used in the fit
        - logf          : log10(f_fit)
        - logP          : log10(P_fit)
        - slope_emp     : empirical slope (free fit)
        - intercept_emp : empirical intercept
        - beta_emp      : empirical beta (= -slope_emp)
        - beta_from_H   : beta from H (or None)
        - a_from_H      : intercept from H-constrained line (or None)
    """
    if fmax is None:
        fmax = fs / 2.0



    # --- 1. Welch PSD ---
    f,Pxx = welch_power_spec(X, fs=500, winsize=5, overlap=0.5)
    
    # --- 2. Select frequency range & go to log-log ---
    mask = (f > 0) & (f >= fmin) & (f <= fmax)
    f_fit = f[mask]
    P_fit = Pxx[mask]

    logf = np.log10(f_fit)
    logP = np.log10(P_fit + 1e-20)

    # --- 3. Unconstrained fit: y = m * logf + c ---
    slope_emp, intercept_emp = np.polyfit(logf, logP, 1)
    beta_emp = -slope_emp  # since y = -beta * logf + a  => slope = -beta


    # --- 5. Plot in log-log space ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    # Data points
    ax.scatter(logf, logP, s=10, alpha=0.7, label="PSD (log-log)")

    # Empirical line
    y_emp = slope_emp * logf + intercept_emp
    ax.plot(logf, y_emp, '--', label=f"Empirical fit (slope={slope_emp:.2f}, β={beta_emp:.2f})")

    # H-based line (if H provided)
    if H is not None:
        
        if type(H) is list:
            
            for iH in H:
                beta_from_H = hurst.hurst_to_slope(iH)        # fractional Gaussian noise relation
    
                a_from_H = np.mean(logP + beta_from_H * logf)
                
                y_H = -beta_from_H * logf + a_from_H
                ax.plot(logf, y_H, '-',
                        label=f"H-based fit (H={iH:.3f}, β={beta_from_H:.2f})")
        else:
            beta_from_H = hurst.hurst_to_slope(H)        # fractional Gaussian noise relation
    
            a_from_H = np.mean(logP + beta_from_H * logf)
            
            y_H = -beta_from_H * logf + a_from_H
            ax.plot(logf, y_H, '-',
                    label=f"H-based fit (H={H:.3f}, β={beta_from_H:.2f})")
            

    ax.set_xlabel("log10(Frequency [Hz])")
    ax.set_ylabel("log10(Power)")
    ax.set_title("PSD in log-log space with linear fits")
    ax.legend()
    plt.tight_layout()

    return {
        "f_fit": f_fit,
        "logf": logf,
        "logP": logP,
        "slope_emp": slope_emp,
        "intercept_emp": intercept_emp,
        "beta_emp": beta_emp,
        "beta_from_H": beta_from_H,
        "a_from_H": a_from_H
    }
