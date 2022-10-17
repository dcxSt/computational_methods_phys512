# To start, model the data as a single Lorenztian and use analytic derivatives. Please use Newton's method (or Levenberg-Marquardt if you prefer) to carry out the fit. What are your best-fit parameters for the amplitude, width, and center? Please parameterize the Lorentzian as d = a/(1+(t-t0)**2/w**2)

import numpy as np
from numpy.linalg import inv


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

def numerical_grad(f,m):
    """Numerically compute the gradiant of f
    
    f : function (think, chi-squared)
    m : model parameters
    """
    return np.vstack([ndiff(f,m,idx) for idx in range(len(m))]).T

def newton_iter_numerical(f,m,d):
    """Returns next iteration of newton's method

    f : function / model
    m : array-like
        Our model parameters, passed to `f` as first arg
    d : measured data
    """
    r=d-A(m,t) # residuals
    Ap=numerical_grad(A,m)
    return m + inv(Ap.T@Ninv@Ap)@Ap.T@r # Ninv's cancel

def mcmc(d,f,m0,cov,Ninv,nstep,step_size,saveas="mcmc.npy"):
    """Computes MCMC of chi-squared given our model A(m,t)

    Parameters
    ----------
    d : np.ndarray
        data
    A : function
        Evalueates the model, takes args (m,t)
    m0 : array-like
        Starting model parameters
    t : np.ndarray
        Times at which to evaluate LASER beam signal
    cov : np.ndarray
        Covariance matrix
    sigma : float
        The estimated noise.
    nstep : int
        Number of MCMC steps.
    step_size : float
        Scaling factor, positive float.

    Returns
    -------
    np.ndarray
        A trace of all parameters used along the chain.
    np.ndarray 
        A trace of all chi-squared values computed along the way. 
        Shape = (nparams, nsteps)
    """
    # Define chi-squared function
    def chi_squared(m0):
        return (d-A(m0,t)).T@(d-A(m0,t))/sigma**2
    # Take Cholesky decomposition to speed things up
    cov_chol = np.linalg.cholesky(cov)
    # Initiate model parameter tracer vectors
    params_trace = np.zeros((len(m0),nstep))
    chisq_trace  = np.zeros(nstep)
    # Compute chisquared
    params_trace[:,0] = m0.copy()
    chisq_trace[0]    = chi_squared(m0)
    # Main loop, wonder around a bit
    for idx in range(1,nstep):
        # Update param
        randvec = np.random.normal(size=m0.shape)
        m = params_trace[:,idx-1] + step_size*cov_chol@randvec
        # Compute accept probability
        chisq = chi_squared(m)
        delta_chisq = chisq - chisq_trace[idx-1]
        # Acceptance probability, determine whether to accept step
        p = np.exp(-0.5*delta_chisq)
        if np.random.rand() < p:
            # Update parameters
            params_trace[:,idx] = m
            chisq_trace[idx]    = chisq
        else:
            # Stay put
            params_trace[:,idx] = params_trace[:,idx-1]
            chisq_trace[idx]    = chisq_trace[idx-1]
    return params_trace, chisq_trace
        


if __name__=="__main__":
    print("dummy")    




