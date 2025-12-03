# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 11:08:29 2025

@author: Guillaume
"""

import numpy as np
import scipy as sp




def fakeEEGsignal(length=240,fs=500,autocorr=0.95,sigma_g=2,sigma_n=2,osc=dict(alpha=[10,1,0.75])):
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
    
    return fake_eeg, norm_noise, aper_component, per_components

def fake_osc(length=240,fs=500,fr=10,amplitude=1,rand_phase=0.75):
    
    t_osc = np.cumsum(np.random.normal(loc=1,scale=rand_phase,size=length*fs))
    osc = np.sin(t_osc/fs*2*np.pi*fr+np.random.uniform()*2*np.pi)*amplitude
    
    return osc

