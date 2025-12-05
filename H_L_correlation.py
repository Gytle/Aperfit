import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

import simul_eeg
import spectrum
import hurst
import Lorentzian_fit


#simulating different fakeeeg without oscillations (only aperiodic component)
def corr_sim():
    results = []
    H_vals = []
    fc_vals = []
    autocorr_values = np.linspace(0.0, 0.99, 100)
    n = 0

    for i in autocorr_values:
        fake_eeg, norm_noise, aper_component, per_component = simul_eeg.fakeEEGsignal(autocorr= i, sigma_n= 0, osc= None)
        #PSD
        fr, pxx = spectrum.welch_power_spec(fake_eeg) 
        #Lorentzian
        f_fit, P_fit, L_fit, A_fit, fc_fit = Lorentzian_fit.fit_psd_lorentzian(fr, pxx, plot=False)
        #DFA
        alpha, scales, fluct = hurst.DFA(fake_eeg)

        #correlations
        H_vals.append(alpha)
        fc_vals.append(fc_fit) 

        
        print(n)
        n = n+1

    H_array = np.array(H_vals)
    fc_array = np.array(fc_vals)

    res_hurst = linregress(H_array, np.log10(fc_array))

    results = {
        "mean_H": np.mean(H_array),
        "mean_fc": np.mean(fc_array),
        "slope": res_hurst.slope,
        "intercept": res_hurst.intercept,
        "R2": res_hurst.rvalue**2
    }

    plt.figure(figsize=(6,4))
    plt.scatter(H_array, np.log10(fc_array), color='blue', label='Data points')

    line_x = np.linspace(np.nanmin(H_array), np.nanmax(H_array), 100)
    line_y = res_hurst.slope * line_x + res_hurst.intercept
    plt.plot(line_x, line_y, 'k--', label=f'Fit: log10(fc) = {res_hurst.slope:.3f}*H + {res_hurst.intercept:.3f}')

    plt.text(np.nanmean(H_array), np.nanmean(np.log10(fc_array)),
             f'log10(fc) = {res_hurst.slope:.3f}*H + {res_hurst.intercept:.3f}',
             fontsize=10, color='red')

    plt.xlabel('Hurst exponent')
    plt.ylabel('log10(Lorentz Fc)')
    plt.title('Correlation Hurst vs Lorentz Fc')
    plt.legend()
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame([results]) 
    print(df)
    return df