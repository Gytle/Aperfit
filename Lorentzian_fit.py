
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from genfakeeeg import fakeEEGsignal
from genfakeeeg import fake_osc
from genfakeeeg import plot_power_spec


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

fake_eeg, norm_noise, aper_component, per_components = fakeEEGsignal(plot=True)
osc = fake_osc()
fr,pxx = plot_power_spec(fake_eeg)
f_fit, P_fit, L_fit, A_fit, fc_fit = fit_psd_lorentzian(fr, pxx)