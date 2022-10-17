import numpy as np
from numpy.linalg import inv
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
    return (f(x+step_h) - f(x-step_h))/(2*dx)

#def numerical_grad(f,m):
#    """Numerically compute the gradiant of f
#    
#    f : function (think, chi-squared)
#    m : parameters (think, model params)
#    """
#    return np.vstack([ndiff(f,m,idx) for idx in range(len(m))]).T

def newton_iter_numerical(f,m):
    """Returns next iteration of newton's method

    Assumes noise matrix is diagonal

    f : function / model
    m : array-like
        Our model parameters, passed to `f` as first arg
    """
    r=f(m)
    fp=numerical_grad(f,m) # f prime
    return m - fp*0.01 #inv(fp.T@fp)@fp.T@r # Ninv's cancel if diagonal



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


def numerical_grad(f,m):
    h=2.0e-08
    grad=np.zeros(len(m))
    print(f"DEBUG: len m={len(m)}")
    print(f"DEBUG: f(m)={f(m)}")
    for xi in range(len(m)):
        print(f"\nDEBUG: xi={xi}")
        dm = np.zeros(len(m))
        dm[xi] = h*max(abs(m[xi]),1.0e-2) # make sure it's the right scale
        ### DEBUG
        print(f"DEBUG: |dm|={h*max(m[xi],1.0e-2)}")
        print(f"DEBUG: f(m+dm)={f(m+dm)}, f(m-dm)={f(m-dm)}")
        dfdxi=(f(m+dm)-f(m-dm))/(2*h*max(abs(m[xi]),1.0e-2))
        print(f"DEBUG: df/dxi={dfdxi}")
        grad[xi]=dfdxi
    return grad


def scale_params(m,m0):
    # scale the params
    return m/m0

def unscale_params(m,m0_true):
    return m*m0_true


def numerical_grad_chisq(get_chisq,pars,spec,errs):
    """wrapper for numerical grad"""
    f=lambda m:get_chisq(m,spec,errs)
    return numerical_grad(f,pars)

def get_chisq(pars,spec,errs):
    """Evaluate chi-squared"""
    model=get_spectrum(pars)
    model=model[:len(spec)]
    resid=spec-model
    return np.sum((resid/errs)**2) # retuern chi-squared

def get_chisq_prime(pars_scaled,spec,errs,m0):
    pars=pars_scaled*m0
    model=get_spectrum(pars)
    model=model[:len(spec)]
    resid=spec-model
    return np.sum((resid/errs)**2) # return chi-squared


#pars=np.asarray([60,0.02,0.1,0.05,2.00e-9,1.0])
pars=np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])
m0=pars.copy() # original starting place
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3]);
chisq=get_chisq(pars,spec,errs)
print("Our initial chisq is ",chisq)#," for ",len(resid)-len(pars)," degrees of freedom.")


# Optimize chi-squared by newton iterations
chisq_arr=[get_chisq(pars,spec,errs)]
stepsize=0.01
for i in range(30):
    print(f"DEBUG: i={i}")
    # Take Cholesky decomposition to speed things up
    #pars=newton_iter_numerical(f=lambda x:get_chisq(x,spec,errs),m=pars)
    # Regular gradient descent
    grad=numerical_grad_chisq(get_chisq,pars,spec,errs)
    print(f"grad={grad}")
    print(f"before pars={pars}")
    pars-=grad*stepsize
    print(f"pars={pars}")


    chisq_arr.append(get_chisq(pars,spec,errs))

print("INFO:chisq array")
for i in chisq_arr: print(i)
    














