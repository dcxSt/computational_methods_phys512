# To start, model the data as a single Lorenztian and use analytic derivatives. Please use Newton's method (or Levenberg-Marquardt if you prefer) to carry out the fit. What are your best-fit parameters for the amplitude, width, and center? Please parameterize the Lorentzian as d = a/(1+(t-t0)**2/w**2)

import numpy as np
from numpy.linalg import inv

dta=np.load("sidebands.npz")
t=dta['time']
d=dta['signal']

def A(m,t):
    """assumes m is the three vector m=(a,t0,w), 
    or a matrix with 3 columns"""
    a,t0,w=m.T
    return a/(1+((t-t0)/w)**2)

def gradA(m,t):
    """assumes m is the three vector m=(a,t0,w), or mat w/ three cols"""
    a,t0,w=m.T
    dAda=1/(1+((t-t0)/w)**2)
    dAdt0=2*a*(t-t0)/(w**2*(1+((t-t0)/w)**2)**2)
    dAdw=2*a*(t-t0)**2/(w**3*(1+((t-to)/w)**2)**2)
    return np.vstack([dAda,dAdt,dAdw]).T

def netwton_iter(m,t,d):
    """Returns next iteration of newton's method"""
    r=A(m,t)-d # residuals
    Ap=gradA(m,t)
    Ninv=np.identity(len(m)) # Our noise model is pretty simple
    return m + inv(Ap.T@Ninv@Ap)@Ap.T@Ninv@r





