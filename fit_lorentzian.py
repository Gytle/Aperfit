# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 12:54:53 2025

@author: Guillaume
"""

import numpy as np
import scipy as sp
from hurst import DFA
from spectrum import welch_power_spec
from hjorth import hjorth_parameters



def lorentzian(f,A,Fc):
    return A/(1+(f/Fc)**2)
    
def find_Fc_hurst(signal,fs):
    alpha,_,_ = DFA(signal)
    
    Fc = (10**(-5.5*np.log10(alpha)-2.25))*fs
    
    return Fc
 
def find_Fc_hjorth(signal,fs):
    act,mob,cx = hjorth_parameters(signal,sampling_rate=fs)
    
    Fc = (10**(-2.959*np.sqrt(np.log10(cx))+0.369))*fs
    
    return Fc

def fit_lorentzian(signal,fs,method='hjorth'):
    
    if method=='hjorth':
        Fc = find_Fc_hjorth(signal,fs=fs)
    elif method=='hurst':
        Fc = find_Fc_hurst(signal,fs=fs)
        
    
    fr,pxx = welch_power_spec(signal,fs)
    
    fitted_spec = lorentzian(fr,1,Fc)
    
    idx_lowfr = (fr<0.05*np.max(fr))
    idx_highfr = (fr>0.95*np.max(fr))
    
    l_pxx = 10*np.log10(pxx)
    l_fitted_spec = 10*np.log10(fitted_spec)
    
    # A = (np.nanmedian(pxx[idx_highfr])-np.nanmedian(pxx[idx_lowfr]))/(np.nanmedian(fitted_spec[idx_highfr])-np.nanmedian(fitted_spec[idx_lowfr]))
    
    
    A = (np.nanmedian(l_pxx[idx_highfr])-np.nanmedian(l_pxx[idx_lowfr]))/(np.nanmedian(l_fitted_spec[idx_highfr])-np.nanmedian(l_fitted_spec[idx_lowfr]))
    
    l_fitted_spec = l_fitted_spec*A
    
    
    # fitted_spec = lorentzian(fr,A,Fc)
    # l_fitted_spec = 10*np.log10(fitted_spec)
    
    intercept = np.nanmedian(l_pxx[idx_lowfr])-np.nanmedian(l_fitted_spec[idx_lowfr])
    
    
    fitted_spec = l_fitted_spec+intercept
    
    return fitted_spec
    

