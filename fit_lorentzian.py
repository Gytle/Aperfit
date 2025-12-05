# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 12:54:53 2025

@author: Guillaume
"""

import numpy as np
import scipy as sp
from hurst import DFA
from spectrum import welch_power_spec



def lorentzian(f,A,Fc):
    return A/(1+(f/Fc)**2)
    
def find_Fc(signal):
    alpha,_,_ = DFA(signal)
    
    Fc = 10**(-3.5*alpha+3.5)
    
    return Fc

def fit_lorentzian(signal,fs):
    
    Fc = find_Fc(signal)
    
    fr,pxx = welch_power_spec(signal,fs)
    
    fitted_spec = lorentzian(fr,1,Fc)
    
    idx_lowfr = (fr<1)
    idx_highfr = (fr>0.95*np.max(fr))
    
    l_pxx = 10*np.log10(pxx)
    l_fitted_spec = 10*np.log10(fitted_spec)
    
    # A = (np.nanmedian(pxx[idx_highfr])-np.nanmedian(pxx[idx_lowfr]))/(np.nanmedian(fitted_spec[idx_highfr])-np.nanmedian(fitted_spec[idx_lowfr]))
    
    
    A = (np.nanmedian(l_fitted_spec[idx_highfr])-np.nanmedian(l_fitted_spec[idx_lowfr]))/(np.nanmedian(l_pxx[idx_highfr])-np.nanmedian(l_pxx[idx_lowfr]))
    
    fitted_spec = lorentzian(fr,A,Fc)
    l_fitted_spec = 10*np.log10(fitted_spec)
    
    intercept = np.nanmedian(l_pxx[idx_lowfr])-np.nanmedian(l_fitted_spec[idx_lowfr])
    
    
    fitted_spec = l_fitted_spec+intercept
    
    return fitted_spec
    

