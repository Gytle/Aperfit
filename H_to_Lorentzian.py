


from simul import simul_eeg
from plot import spectrum
from utils import hurst

def H_to_L():
    fake_eeg, norm_noise, aper_component, per_component = simul_eeg.fakeEEGsignal()
    alpha, scales, fluct = hurst.DFA(fake_eeg)
    l_coeff = 10**(-3.5*alpha +3.5)
    return l_coeff

def L_fit():
    
