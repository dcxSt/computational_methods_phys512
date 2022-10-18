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
    #return np.array([ndiff(f,x,idx) for idx in range(len(x))])
    return np.vstack([ndiff(f,x,idx) for idx in range(len(x))]).T


# used by chi_squared
def get_spectrum(pars,lmax=3000):
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


def get_chisq(pars,spec,errs):
    """Evalueate chi-sq"""
    model=get_spectrum(pars)
    model=model[:len(spec)]
    resid=spec-model
    return np.sum((resid/errs)**2)

def get_chisq_p(s_pars,spec,errs,m0):
    pars=s_pars*m0 # scaled parameters are pars/m0=:s_pars
    model=get_spectrum(pars)
    model=model[:len(spec)]
    resid=spec-model
    return np.sum((resid/errs)**2) # return chi-sq

# Initial guess for the parameters
m0=np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])
npars=len(m0) # number of parameters
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3])

pars=m0.copy()
chisq=get_chisq(pars,spec,errs)

def f(s_pars):
    return get_chisq_p(s_pars,spec,errs,m0)

def get_spectrum_p(s_pars):
    model=get_spectrum(s_pars*m0) # strange python scope allows me to m0
    model=model[:len(spec)]
    return model


# def newton_iter_numerical(get_spectrum_p,pars,spec,m0=m0):
#     s_pars=pars/m0
#     model=get_spectrum_p(s_pars)
#     r=spec-model
#     grad_p=numerical_grad(get_spectrum_p,s_pars)
#     s_pars = s_pars + pinv(grad_p.T@grad_p)@grad_p.T@r # s_ is for scaled
#     #grad=grad_p*m0 # _p suffix is for prime
#     print(f"DEBUG: grad_p computed, shape={grad_p.shape}")
#     print(f"\ts_pars={s_pars}")
#     print(f"")
#     return s_pars*m0#pars + inv(grad.T@grad)@grad.T@r # Ninv's cancel

def newton_iter(A,m,d,m0):
    """Iterate newton's method to minimize (A(m)-spec).T@Ninv@(A(m)-spec)

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

    Returns
    -------
        Next iteration of best-guess of optimal parameters. 
    """
    # This scaled model function uses it's environment, 
    # like a closure in other languages
    def A_scaled(m_scaled):
        model=A(m_scaled*m0) 
        model=model[:len(d)] # trunkate model to fit data
        return model
    m_scaled=m/m0               # Scale the input appropriately
    model=A_scaled(m_scaled)    # Evaluate the model
    resid=spec-model            # Compute residuals
    grad_scaled=numerical_grad(A_scaled,m_scaled) # Scaling important for this step
    m_scaled += pinv(grad_scaled.T@grad_scaled)@grad_scaled.T@resid # Update
    return m_scaled*m0          # Re-scale and return
    
#def newton_iter2(get_spectrum,pars,spec,m0):
#    model=get_spectrum(pars)[:len(spec)]
#    r=spec-model
#    grad=numerical_grad(get_spectrum,pars)
#    print(f"DEBUG: grad computed, shape={grad.shape}")
#    return pars + inv(grad.T@grad)@grad.T@r # Ninv's cancel


print(f"DEBUG: chi-squared evaluates to {chisq}")

## lets do some tests
#print("DEBUG: taking partial derivatives")
#for idx in range(npars):
#    partial=ndiff(f,pars/m0,idx)
#    print(f"DEBUG: partial_{idx} = {partial}")



print("INFO: Newton iter")
print(f"\tchisq={get_chisq(pars,spec,errs)}")
for i in range(20):
    print(f"\nDEBUG: iteration {i}/20")
    #pars = newton_iter_numerical(get_spectrum_p,pars,spec,m0)
    pars = newton_iter(get_spectrum,pars,spec,m0)
    print(f"\tchisq={get_chisq(pars,spec,errs)}")

# stepsize=1.0e-6
# print("INFO: Testing Nabla")
# for i in range(10):
#     nabla=numerical_grad(f,pars/m0)
#     print("DEBUG: nabla",nabla)
#     print("\tstepsize*nabla*m0",stepsize*nabla*m0)
#     pars-=stepsize*nabla*m0
#     print("\tpars",pars)
#     print(f"DEBUG: chisq={get_chisq(pars,spec,errs)}")
















