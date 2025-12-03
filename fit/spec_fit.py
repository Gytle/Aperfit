# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 15:36:32 2025

@author: Guillaume
"""

import numpy as np
import scipy as sp


def aperfunc(f,A,k,b):
    
    y = 10**A*(1/(k+f**b))
    
    return y

def lorentz(f,A,tau):
    
    y = A/(1+(2*np.pi*tau*f)**2)
    
    return y