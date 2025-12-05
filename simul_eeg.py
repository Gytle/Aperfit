# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 11:08:29 2025

@author: Guillaume
"""

import numpy as np

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