# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 16:16:27 2025

@author: Guillaume
"""
import numpy as np
from scipy.signal import welch
from scipy.optimize import curve_fit
from hurst import DFA

# ---------- 1. Example PSD (replace this with your own data) ----------

def compute_psd(signal, fs, nperseg=2048):
    """Compute PSD via Welch."""
    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend='constant',
        scaling='density'
    )
    return freqs, psd

# ---------- 2. Knee-ized 1/f model with fixed beta ----------

def log_psd_model(f, log10_A, log10_fknee, beta):
    """
    Model log10 P(f) = log10 A - log10[1 + (f/f_k)^beta]
    Parameters:
        f        : frequencies (Hz)
        log10_A  : log10 amplitude at low f
        log10_fknee : log10 knee frequency
        beta     : fixed slope exponent (>0)
    """
    A = 10**log10_A
    fknee = 10**log10_fknee
    return np.log10(A) - np.log10(1.0 + (f / fknee)**beta)

def fit_knee_amplitude(freqs, psd, beta, fmin=None, fmax=None, 
                       initial_log10_A=None, initial_log10_fknee=None):
    """
    Fit amplitude and knee for a given beta.
    
    Parameters
    ----------
    freqs : array
        Frequencies (Hz).
    psd   : array
        Power spectral density (linear units).
    beta  : float
        Fixed slope exponent.
    fmin, fmax : float or None
        Frequency range to use for the fit.
    initial_log10_A, initial_log10_fknee : float or None
        Optional initial guesses for parameters (log10 scale).
        
    Returns
    -------
    params : dict
        {"A": ..., "fknee": ..., "log10_A": ..., "log10_fknee": ...}
    """
    # 1. Select fitting range
    mask = np.ones_like(freqs, dtype=bool)
    if fmin is not None:
        mask &= freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax

    f_fit = freqs[mask]
    P_fit = psd[mask]

    # Avoid zeros / negatives for log
    P_fit = np.maximum(P_fit, np.finfo(float).tiny)
    logP = np.log10(P_fit)

    # 2. Initial guesses
    if initial_log10_A is None:
        # rough guess: mean of top 10% lowest freq powers
        low_mask = f_fit < (fmin + 0.1*(fmax - fmin)) if (fmin is not None and fmax is not None) else slice(0, len(f_fit)//5)
        A0 = np.median(P_fit[low_mask])
        initial_log10_A = np.log10(A0)

    if initial_log10_fknee is None:
        # rough guess: median of fitting freqs
        fknee0 = np.median(f_fit)
        initial_log10_fknee = np.log10(fknee0)

    p0 = [initial_log10_A, initial_log10_fknee]

    # 3. Fit with curve_fit
    def model_to_fit(f, log10_A, log10_fknee):
        return log_psd_model(f, log10_A, log10_fknee, beta)

    popt, pcov = curve_fit(model_to_fit, f_fit, logP, p0=p0)

    log10_A_hat, log10_fknee_hat = popt
    A_hat = 10**log10_A_hat
    fknee_hat = 10**log10_fknee_hat

    return {
        "A": A_hat,
        "fknee": fknee_hat,
        "log10_A": log10_A_hat,
        "log10_fknee": log10_fknee_hat,
        "cov": pcov,
    }

# ---------- 3. Example usage ----------

import mne
import os
import matplotlib.pyplot as plt
import GED
from hurst import DFA
from hjorth import hjorth_parameters

# datapath = 'C:\\Users\\Guillaume\\Downloads\\sub-032301_2\\sub-032301\\RSEEG'
datapath = 'C:\\Users\\Guillaume\\Downloads\\sub-032301\\sub-032301'
# raw = mne.io.read_raw_brainvision(os.path.join(datapath,'sub-032301.vhdr'))
raw = mne.io.read_raw_eeglab(os.path.join(datapath,'sub-032301_EO.set'))






signal = raw.get_data()



sig_len = 60*5*250
win_len = 60*250

signal_red = signal[:,:sig_len]

tp = 10000

# list_hurst = np.zeros((signal_red.shape[0],tp))
list_hjorth = np.zeros((signal_red.shape[0],tp))
for iCh in np.arange(signal_red.shape[0]):
    for iT in np.arange(tp):
        tmp_sig = signal_red[iCh,np.arange(win_len)+iT]
        
        # alpha,_,_ = DFA(tmp_sig)
        act,mob,complx = hjorth_parameters(tmp_sig)
        
        # list_hurst[iCh,iT] = alpha
        list_hjorth[iCh,iT] = np.log(complx).
        
        
        
R = signal_red

covR = R@R.T/(R.shape[1]-1)
covR = 0.5 * (covR + covR.T)
covR = GED.regularize(covR)

S = list_hjorth

covS = S@S.T/(S.shape[1]-1)
covS = 0.5 * (covS + covS.T)

W,A,W_norm,var_exp = GED.GED(covR,covS)



# Fake example: mixture of Lorentzians to emulate 1/f^beta
rng = np.random.default_rng(0)
fs = 250.0  # Hz
N = 60 * fs  # 60 seconds
t = np.arange(N) / fs
# some random signal; replace with real EEG channel
signal = rng.standard_normal(N)

# Suppose you've already estimated H (e.g. via DFA) and decided:
H = 0.8
beta = 2 * H  # <- your chosen mapping

freqs, psd = compute_psd(signal, fs)

# Fit in, say, 2–40 Hz
fmin, fmax = 2.0, 40.0
fit = fit_knee_amplitude(freqs, psd, beta=beta, fmin=fmin, fmax=fmax)
A_hat = fit["A"]
fknee_hat = fit["fknee"]

print(f"Estimated A = {A_hat:.3e}")
print(f"Estimated fknee = {fknee_hat:.2f} Hz")

# Plot data and fitted model
plt.figure(figsize=(7,5))
plt.loglog(freqs, psd, label="PSD")

# model over full freq range
model_logP = log_psd_model(freqs, fit["log10_A"], fit["log10_fknee"], beta)
model_P = 10**model_logP
plt.loglog(freqs, model_P, "--", label=f"fit (β={beta:.2f})")

plt.axvline(fknee_hat, color='k', linestyle=':', label="fknee")
plt.xlim(0.5, fs/2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()