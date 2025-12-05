# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 11:08:29 2025

@author: Guillaume
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import hurst
import spectrum
import simul_eeg


plt.rcParams['svg.fonttype'] = 'none'

fs = 500

fake_eeg, norm_noise, aper_component, per_components = simul_eeg.fakeEEGsignal(
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
axes[0].set_ylim([-20,20])

# 2) Aperiodic (fractal) component
axes[1].plot(t5, aper_component[mask])
axes[1].set_ylabel("Amplitude")
axes[1].set_title("Aperiodic component")
axes[1].set_ylim([-20,20])

# 3) White Gaussian noise
axes[2].plot(t5, norm_noise[mask])
axes[2].set_ylabel("Amplitude")
axes[2].set_title("Gaussian noise")
axes[2].set_ylim([-20,20])

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
axes[3].set_ylim([-20,20])

plt.tight_layout()
plt.show()

fs = 500
fake_eeg, norm_noise, aper_component, per_components = simul_eeg.fakeEEGsignal(
    length=240,
    fs=fs,
    autocorr=0.95,
    sigma_g=2,
    sigma_n=2,
    osc=dict(alpha=[10,1,0.75]),
    plot=True)



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
    H_comp  = [H_fromRS_list[idx],H_fromDFA_list[idx],H_fromWav_list[idx]]

    results = spectrum.fit_psd_line_with_H(
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
    H_comp  = [H_fromRS_list[idx],H_fromDFA_list[idx],H_fromWav_list[idx]]

    _ = spectrum.fit_psd_line_loglog(
        sig,
        fs=fs,
        H=H_comp,
        fmin=1.0,
        fmax=fs/2,
        ax=ax
    )
    ax.set_title(f"log–log PSD – {sig_lab}")

plt.tight_layout()


