# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 11:08:29 2025

@author: Guillaume
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from utils import hurst


def fakeEEGsignal(length=240, fs=500, autocorr=0.99, sigma_g=1, sigma_n=1,
                  osc=dict(alpha=[10, 2, 0.75]), plot=False):
    """ Function to generate a fake EEG signal

    Parameters
    ----------
    length : float
        Length of the signal in seconds (default 240 s).
    fs : float
        Sampling rate (default 500 Hz).
    autocorr : float
        AR(1) coefficient of the aperiodic/fractal component (default 0.95–0.99).
    sigma_g : float
        Amplitude (std) of the aperiodic driving noise.
    sigma_n : float
        Amplitude (std) of the random Gaussian noise.
    osc : dict or None
        Dict of oscillatory components. Each entry: [frequency, amplitude, phase_jitter].
    plot : bool
        If True, plot PSDs of all components.
    """

    t_gen = np.arange(length * fs) / fs

    # ---- Periodic components ----
    if osc is not None:
        per_components = []
        for name in osc:   # e.g., 'alpha'
            tmp_osc = osc[name]
            tmp_rhythm = fake_osc(length=length, fs=fs,
                                  fr=tmp_osc[0],
                                  amplitude=tmp_osc[1],
                                  rand_phase=tmp_osc[2])
            per_components.append(tmp_rhythm)

        per_components = np.stack(per_components)   # shape: (n_osc, N)
    else:
        # keep 2D shape so np.sum(per_components, axis=0) works the same
        per_components = np.zeros((1, t_gen.size))

    # ---- Aperiodic component (AR(1)-like) ----
    rand_noise = np.random.normal(loc=0, scale=sigma_g, size=t_gen.size)
    aper_component = np.zeros(t_gen.size)
    for iX in range(1, t_gen.size):
        aper_component[iX] = autocorr * aper_component[iX - 1] + rand_noise[iX]

    # ---- White noise ----
    norm_noise = np.random.normal(loc=0, scale=sigma_n, size=t_gen.size)

    # ---- Full signal ----
    fake_eeg = np.sum(per_components, axis=0) + aper_component + norm_noise

    # ---- Plot power spectra of components ----
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        plot_power_spec(norm_noise, fs=fs, ax=axes[0, 0])
        axes[0, 0].set_title("Gaussian noise")

        plot_power_spec(aper_component, fs=fs, ax=axes[0, 1])
        axes[0, 1].set_title("Aperiodic component")

        if per_components.ndim > 1:
            for iOsc in range(per_components.shape[0]):
                plot_power_spec(per_components[iOsc, :], fs=fs, ax=axes[1, 0])
        else:
            plot_power_spec(per_components[0, :], fs=fs, ax=axes[1, 0])
        axes[1, 0].set_title("Oscillations")

        plot_power_spec(fake_eeg, fs=fs, ax=axes[1, 1])
        axes[1, 1].set_title("Full fake EEG")

        plt.tight_layout()
        plt.show()

    return fake_eeg, norm_noise, aper_component, per_components


def fake_osc(length=240, fs=500, fr=10, amplitude=1, rand_phase=0.75):
    """ Function to generate a fake oscillation

    Parameters
    ----------
    length : float
        Length of the signal in seconds.
    fs : float
        Sampling rate.
    fr : float
        Oscillation frequency (Hz).
    amplitude : float
        Oscillation amplitude.
    rand_phase : float
        Std of random phase jitter.
    """
    t_osc = np.cumsum(
        np.random.normal(loc=1, scale=rand_phase, size=length * fs)
    )
    osc = np.sin(t_osc / fs * 2 * np.pi * fr + np.random.uniform() * 2 * np.pi) * amplitude
    return osc


def plot_power_spec(X, fs=500, winsize=5, overlap=0.5, ax=None):
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
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

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

    ax.plot(fr, 10 * np.log10(pxx))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_xlim(0, fs / 2)

    return ax

fs = 500

fake_eeg, norm_noise, aper_component, per_components = fakeEEGsignal(
    length=240,
    fs=fs,
    autocorr=0.95,
    sigma_g=2,
    sigma_n=2,
    osc=dict(alpha=[10, 1, 0.75])
)

t = np.arange(fake_eeg.size) / fs
mask = t < 5
t5 = t[mask]

fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
fig.suptitle("Fake EEG and its Components (first 5 s)", fontsize=14)

# 1) Full fake EEG
axes[0].plot(t5, fake_eeg[mask])
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Fake EEG")

# 2) Aperiodic (fractal) component
axes[1].plot(t5, aper_component[mask])
axes[1].set_ylabel("Amplitude")
axes[1].set_title("Aperiodic component")

# 3) White Gaussian noise
axes[2].plot(t5, norm_noise[mask])
axes[2].set_ylabel("Amplitude")
axes[2].set_title("Gaussian noise")

# 4) Oscillatory components
if per_components is not None:
    # per_components has shape (n_osc, n_samples)
    for i in range(per_components.shape[0]):
        axes[3].plot(t5, per_components[i, mask], label=f"osc #{i+1}")
    axes[3].legend(loc="upper right")
    axes[3].set_title("Oscillatory components")
else:
    axes[3].text(0.5, 0.5, "No oscillatory components",
                 ha="center", va="center", transform=axes[3].transAxes)

axes[3].set_xlabel("Time (s)")
axes[3].set_ylabel("Amplitude")

plt.tight_layout()
plt.show()

fs = 500
fake_eeg, norm_noise, aper_component, per_components = fakeEEGsignal(
    length=240,
    fs=fs,
    autocorr=0.95,
    sigma_g=2,
    sigma_n=2,
    osc=dict(alpha=[10,1,0.75]),
    plot=True)

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

    # --- 1. Welch PSD ---
    nperseg = int(winsize * fs)
    noverlap = int(overlap * nperseg)

    f, Pxx = sp.signal.welch(
        X,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg
    )

    # --- 2. Select frequency range for fitting (avoid f=0) ---
    mask = (f > 0) & (f >= fmin) & (f <= fmax)
    f_fit = f[mask]
    P_fit = Pxx[mask]

    logf = np.log10(f_fit)
    logP = np.log10(P_fit)

    # --- 3. Slope from H and intercept from least-squares ---
    beta = 2*H          # fractional Gaussian noise
    slope = -beta             # log10 P ~ intercept + slope * log10 f

    # best intercept: mean of (logP - slope*logf)
    intercept = np.mean(logP - slope * logf)

    # reconstruct fitted line in log10 space
    logP_fit = intercept + slope * logf

    # convert to linear power and then to dB (to overlay on your dB PSD)
    P_fit_lin = 10**logP_fit
    P_fit_db = 10 * np.log10(P_fit_lin)

    # full PSD in dB (for plotting)
    Pxx_db = 10 * np.log10(Pxx + 1e-20)

    # --- 4. Plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    # Original PSD
    ax.plot(f, Pxx_db, label="PSD (Welch)")

    # Fitted 1/f^beta line (from H)
    ax.plot(f_fit, P_fit_db, '--', label=f"1/f^β fit from H (H={H:.3f})")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_xlim(0, fs / 2)
    ax.legend()
    ax.set_title("PSD with 1/f fit from Hurst exponent")

    return {
        "f": f,
        "Pxx": Pxx,
        "f_fit": f_fit,
        "P_fit_db": P_fit_db,
        "beta": beta,
        "slope": slope,
        "intercept": intercept
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
    nperseg  = int(winsize * fs)
    noverlap = int(overlap * nperseg)

    f, Pxx = sp.signal.welch(
        X,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg
    )

    # --- 2. Select frequency range & go to log-log ---
    mask = (f > 0) & (f >= fmin) & (f <= fmax)
    f_fit = f[mask]
    P_fit = Pxx[mask]

    logf = np.log10(f_fit)
    logP = np.log10(P_fit + 1e-20)

    # --- 3. Unconstrained fit: y = m * logf + c ---
    slope_emp, intercept_emp = np.polyfit(logf, logP, 1)
    beta_emp = -slope_emp  # since y = -beta * logf + a  => slope = -beta

    # --- 4. Constrained fit from H, if provided ---
    beta_from_H = None
    a_from_H = None
    if H is not None:
        beta_from_H = 2*H        # fractional Gaussian noise relation
        # y = -beta_from_H * logf + a_H
        # => a_H = mean( logP + beta_from_H * logf )
        a_from_H = np.mean(logP + beta_from_H * logf)

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

# Hurst exponents per component

fs = 500
osc_sum = np.sum(per_components, axis=0)  # shape (N,)

signals = [fake_eeg, norm_noise, aper_component, osc_sum]
labels  = ["Full fake EEG", "Gaussian noise", "Aperiodic", "Oscillations"]

H_fromRS_list   = []
H_fromDFA_list   = []
H_fromWav_list   = []
info_list = []

# Compute H for each component
for sig in signals:
    H, info = hurst.hurst_rs(sig)
    H_fromRS_list.append(H)
    alpha, scales, fluct = hurst.DFA(sig)
    H_fromDFA_list.append(alpha)
    H, scales, variances, slope = hurst.hurst_wavelet(sig)
    H_fromWav_list.append(H)
    info_list.append(info)

# ---- Tiled R/S plots ----
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for idx, ax in enumerate(axes.ravel()):
    sig_label = labels[idx]
    H = H_fromRS_list[idx]
    info = info_list[idx]
    n  = info["n"]
    RS = info["RS"]

    logn  = np.log(n)
    logRS = np.log(RS)
    fit   = info["intercept"] + info["slope"] * logn

    ax.scatter(logn, logRS, s=10, label="data")
    ax.plot(logn, fit, label=f"fit, H={H:.3f}")
    ax.set_xlabel("log n")
    ax.set_ylabel("log R/S")
    ax.set_title(f"R/S – {sig_label}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()



#1/f^beta fits
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for idx, ax in enumerate(axes.ravel()):
    sig     = signals[idx]
    sig_lab = labels[idx]
    H_comp  = H_fromRS_list[idx]

    results = fit_psd_line_with_H(
        sig,
        fs=fs,
        H=H_comp,
        fmin=1.0,      # avoid 0 Hz
        fmax=fs/2,
        winsize=5,
        overlap=0.5,
        ax=ax
    )

    ax.set_title(f"PSD + 1/f fit – {sig_lab}")

plt.tight_layout()
plt.show()



# log-log fits
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for idx, ax in enumerate(axes.ravel()):
    sig     = signals[idx]
    sig_lab = labels[idx]
    H_comp  = H_list[idx]

    _ = fit_psd_line_loglog(
        sig,
        fs=fs,
        H=H_comp,
        fmin=1.0,
        fmax=fs/2,
        ax=ax
    )
    ax.set_title(f"log–log PSD – {sig_lab}")

plt.tight_layout()


