import numpy as np
from numpy.linalg import inv,pinv
import camb
from matplotlib import pyplot as plt
import time

# Problem specitc chi-squared
# used by chi_squared
def get_spectrum(pars,lmax=3000):
    """Evaluate model at pars"""
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2:]

def get_chisq(pars,spec,errs,A=get_spectrum):
    """Evalueate chi-sq"""
    model=A(pars)
    model=model[:len(spec)]
    resid=spec-model
    return np.sum((resid/errs)**2)

if __name__ == "__main__":
    # Initial guess for the parameters
    m0=np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])
    m1=np.asarray([150,0.07,-0.035,0.06,2.1e-9,3.3])
    npars=len(m0) # number of parameters
    parnames=["Hubble constant H0","Baryon density","Dark matter density",
            "Optical depth","Primordial amplitude","Primordial tilt"]
    # Data
    planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
    ell=planck[:,0]
    spec=planck[:,1]
    errs=0.5*(planck[:,2]+planck[:,3])
    Ninv=1/errs**2 # Inverse diagonal matrix of errors
    
    pars=m0.copy()
    chisq=get_chisq(pars,spec,errs,get_spectrum)

    print(f"INFO: chisq m0={get_chisq(m0,spec,errs,get_spectrum)}")
    print(f"INFO: chisq m1={get_chisq(m1,spec,errs,get_spectrum)}")
    
    
















