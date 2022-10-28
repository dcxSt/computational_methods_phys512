import numpy as np
from numpy.linalg import inv,pinv
import camb
from matplotlib import pyplot as plt
import time

### Numerical derivatives
def ndiff(f,x,idx):
    """Take the partial derivative of f at x wrt to the idx'th argument.

    f : function
        A multivariate function
    x : np.ndarray
        A vector argument, where we evaluate the derivaitve at
    idx : int
        The index of the direction along which we evaluate the partial 
        deriv
    """
    # Select some optimal step sizes
    # We approximate roundoff error epsilon=10.0e-16
    # Highest order term in 2 point derivative TS expansion is o(dx**3)
    # Optimal step size is about 2*sqrt(epsilon) = 2.0e-08
    # (ballpark back of envolope estimate)
    dx = 2.0e-08
    step_h = np.zeros(x.shape)
    step_h[idx] = dx
    # Return the numerical partial derivative of f wrt it's argument at 
    # idx, at x
    #print(f"DEBUG: f(x+-step_h)={f(x+step_h)},{f(x-step_h)}")
    return (f(x+step_h) - f(x-step_h))/(2*dx)


def numerical_grad(f,x):
    """Numerically compute the gradiant of f
    
    f : function (think, chi-squared)
    x : parameters (think, model params)

    returns : ndarray
        Gradiant of f, with dims  len(f(x)) X len(x) 
    """
    return np.vstack([ndiff(f,x,idx) for idx in range(len(x))]).T


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

def newton_iter(A,m,d,m0,Ninv):
    return m + newton_delta(A,m,d,m0,Ninv)

def newton_delta(A,m,d,m0,Ninv):
    """Iterate newton's method to minimize (A(m)-d).T@Ninv@(A(m)-d)

    We don't need the errors because, for our purposes they are 
    diagonal. 

    Parameters
    ----------
    A : function
        The model, takes the vector argument m
    m : ndarray
        Model parameters
    d : ndarray
        The data we are fitting to. Dimensions must match size 
        of output of A(m).
    m0 : ndarray
        To make sure our parameters are more or less optimal scale for 
        the numerical derivatives. The size must match the size of m. 
    Ninv : ndarray
        1d numpy array representing diagonal uncorrelated noise matrix.

    Returns
    -------
        Next iteration of best-guess of optimal parameters. 
    """
    # This scaled model function uses it's environment, 
    # like a closure in other languages
    def A_scaled(m_scaled):
        model=A(m_scaled*m0)
        model=model[:len(d)] # trunkate model to fit data size
        return model
    m_scaled=m/m0               # Scale input appropriately
    model=A_scaled(m_scaled)    # Evaluate model
    resid=d-model               # Compute residuals
    grad_scaled=numerical_grad(A_scaled,m_scaled) # Compute gradiant
    grad=grad_scaled/m0         # Undo normalization scaling
    cov=inv((grad.T*Ninv)@grad) # Covariance matrix
    #print(f"DEBUG: Compare cov matrices to debug get cov func\n\t{cov-get_covariance_matrix(A,m,d,m0,Ninv)}\n")
    return cov@(grad.T*Ninv)@resid

def get_covariance_matrix(A,m,d,m0,Ninv):
    """Get the covariance matrix at m

    Parameters
    ----------
    A : function
        model
    m : ndarray
        model parameters
    m0 : ndarray 
        scaling parameters, usually set to initial guess model params
    Ninv : ndarray
        1d array, represents uncorrelated errors (diagonal matrix).
    
    Returns
    -------
    ndarray
        covariance matrix
    """
    # Scale the model to take numerical derivative
    def A_scaled(m_scaled):
        model=A(m_scaled*m0)
        model=model[:len(d)] # trunkate model to fit data size
        return model
    m_scaled=m/m0 # Normalize to optimize numerical derivative
    grad_scaled=numerical_grad(A_scaled,m_scaled) # Compute gradiant
    grad=grad_scaled/m0 # Undo normalization
    return inv((grad.T*Ninv)@grad) # Return the covariance matrix
    

if __name__ == "__main__":
    # Initial guess for the parameters
    m0=np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])
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
    
    
    print(f"DEBUG: chi-squared evaluates to {chisq}")
    
    ## lets do some tests
    #print("DEBUG: taking partial derivatives")
    #for idx in range(npars):
    #    partial=ndiff(f,pars/m0,idx)
    #    print(f"DEBUG: partial_{idx} = {partial}")
    
    
    
    print("INFO: Newton iter")
    print(f"\tchisq={get_chisq(pars,spec,errs,get_spectrum)}")
    for i in range(8):
        print(f"\nDEBUG: iteration {i+1}/4")
        pars = newton_iter(get_spectrum,pars,spec,m0,Ninv)
        print(f"\tchisq={get_chisq(pars,spec,errs,get_spectrum)}")
        print(f"\tpars={pars}")
    
    # Get the covariance matrix
    cov=get_covariance_matrix(get_spectrum,pars,spec,m0,Ninv)
    sigma_pars=np.sqrt(np.diag(cov))
    print("INFO: The best fit parameters are")
    for parname,param,sigma in zip(parnames,pars,sigma_pars):
        print(f"  {parname}\t{param:.3e} +- {sigma:.1e}")
    
    # Serialize parameters
    print("INFO: Serializing parameters")
    import json
    with open("plank_fit_params.txt","w") as f:
        json.dump({
            "chisq":get_chisq(pars,spec,errs,get_spectrum),
            "parnames":parnames,
            "pars":list(pars),
            "sigma_pars":list(sigma_pars),
            "cov":[[i for i in j] for j in cov]
            },f,indent=2)

















