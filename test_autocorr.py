# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 13:19:31 2025

@author: Guillaume
"""

from simul.simul_eeg import fakeEEGsignal
from utils import hurst
from plot.spectrum import welch_power_spec
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import scipy as sp


def lorentzian(f, A, fc):
    return A / (1 + (f / fc)**2)

def avalanche(f,A,b,k):
    return A / (k + f**b)


def fit_psd_lorentzian(fr, pxx, fmin = 1, fmax = None, plot = True):

    # Frequency range
    mask = fr > 0
    mask &= fr >= fmin
    if fmax is not None:
        mask &= fr <= fmax

    f_fit = fr[mask]
    P_fit = pxx[mask]

    A0 = np.max(P_fit)
    fc0 = f_fit[np.argmax(P_fit)] 

    # Fit
    popt, pcov = curve_fit(
        lorentzian,
        f_fit,
        P_fit,
        p0=[A0, fc0],
        bounds=([0, 0], [np.inf, np.inf])
    )

    A_fit, fc_fit = popt
    L_fit = lorentzian(f_fit, A_fit, fc_fit)

    if plot:
        plt.figure(figsize=(7,4))
        plt.plot(fr, 10*np.log10(pxx), label="PSD (Welch)")
        plt.plot(f_fit, 10*np.log10(L_fit), '--', label=f"Lorentz fit: A={A_fit:.2e}, fc={fc_fit} Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.title("EEG PSD with Lorentzian fit")
        plt.legend()
        plt.show()

    return f_fit, P_fit, L_fit, A_fit, fc_fit

autocorr_param = np.arange(0.80,0.99,0.01)

H_list = []
lorentz_list = []
for iP in np.arange(autocorr_param.size):
    fake_eeg, norm_noise, aper_component, per_components = fakeEEGsignal(autocorr=autocorr_param[iP], sigma_n=0,osc=None)

    alpha, scales, fluct = hurst.DFA(fake_eeg)
    
    f,pxx = welch_power_spec(fake_eeg)
    
    f_fit, P_fit, L_fit, A_fit, fc_fit = fit_psd_lorentzian(f, pxx, plot=True)
    
    H_list.append(alpha)
    lorentz_list.append(fc_fit)
    
res = sp.stats.linregress(np.array(H_list),np.log(np.array(lorentz_list)))
    
plt.figure()
plt.scatter(np.array(H_list),np.log(np.array(lorentz_list)))
plt.plot(np.array(H_list),np.array(H_list)*res.slope+res.intercept,'k--')
plt.text(np.nanmean(np.array(H_list)),np.nanmean(np.log(np.array(lorentz_list))),r'Fc=%02.03f*H+%02.03f'%(res.slope,res.intercept))

print(r'Final relation H x lorentzian fit: Fc=%02.03f*H+%02.03f'%(res.slope,res.intercept))
