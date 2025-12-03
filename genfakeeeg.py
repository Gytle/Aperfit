# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 11:08:29 2025

@author: Guillaume
"""

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

def fakeEEGsignal(length=240,fs=500,autocorr=0.99,sigma_g=1,sigma_n=1,
                  osc=dict(alpha=[10,2,0.75]),plot=False):
    """ Function to generate a fake EEG signal
    
    Length is the length in seconds of the example signal (default length=240 s)
    fs is the sampling rate (default fs=500)
    autocorr is the level of autocorrelation of the aperiodic/fractal component (default autocorr=0.95)
    sigma_g is the amplitude of the aperiodic component (default sigma_g=2)
    sigma_n is the amplitude of the random gaussian noise (default sigma_n=2)
    osc is a dictionnary composed of several oscillatory components. Each component should be a list with three numbers: the frequency, the amplitude and the randomness in the phase
    """
    
    
    t_gen = np.arange(length*fs)/fs
    np.random.normal(loc=0.0, scale=1.0, size=None)
    
    if osc is not None:
        osc_names = osc.keys()
        
        per_components = []
        for iOsc in osc:
            tmp_osc = osc[iOsc]
            tmp_rhythm = fake_osc(length=length,fs=fs,
                                  fr=tmp_osc[0],amplitude=tmp_osc[1],rand_phase=tmp_osc[2])
            
            per_components.append(tmp_rhythm)
        
        per_components = np.stack(per_components)
    else:
        per_components = np.zeros(t_gen.size)
        
        
    rand_noise = np.random.normal(loc=0,scale=sigma_g,size=t_gen.size)
    aper_component = np.zeros(t_gen.size)
    for iX in np.arange(1,t_gen.size):
        aper_component[iX] = autocorr*aper_component[iX-1]+rand_noise[iX]
    
    norm_noise = np.random.normal(loc=0,scale=sigma_n,size=t_gen.size)
    
    fake_eeg = np.sum(per_components,axis=0)+aper_component+norm_noise
    
    if plot:
        plt.figure()
        plt.subplot(2,2,1)
        plot_power_spec(norm_noise)
        plt.title('Gaussian noise')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.subplot(2,2,2)
        plot_power_spec(aper_component)
        plt.title('Aperiodic component')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.subplot(2,2,3)
        if per_components.ndim>1:
            for iOsc in range(per_components.shape[0]):
                plot_power_spec(per_components[iOsc,:])
        else:
            plot_power_spec(per_components)
        plt.title('Oscillations')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.subplot(2,2,4)
        plot_power_spec(fake_eeg)
        plt.title('Full fake EEG')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.show()
    
    return fake_eeg, norm_noise, aper_component, per_components

def fake_osc(length=240,fs=500,fr=10,amplitude=1,rand_phase=0.75):
    """ Function to generate a fake oscillation
    Length is the length in seconds of the example signal (default length=240 s)
    fs is the sampling rate (default fs=500)
    fr is the frequency of the oscillation (default fr=10)
    amplitude is the amplitude of the oscillation (default amplitude=1)
    rand_phase introduce a random jitter of the phase of the oscillation so looks realistic (default rand_phase=0.75)
    """
    
    t_osc = np.cumsum(np.random.normal(loc=1,scale=rand_phase,size=length*fs))
    osc = np.sin(t_osc/fs*2*np.pi*fr+np.random.uniform()*2*np.pi)*amplitude
    
    return osc

def plot_power_spec(X, fs=500, winsize=5, overlap=0.5):
    
    winsize_t = np.int64(winsize*fs)
    overlapsize_t = np.int64(winsize*fs*overlap)
    
    fr,pxx = sp.signal.welch(X, fs=fs, window='hann', nperseg=winsize_t,
                    noverlap=overlapsize_t,
                    nfft=winsize_t)
    
    plt.plot(fr,10*np.log10(pxx))

