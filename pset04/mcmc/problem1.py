# To start, model the data as a single Lorenztian and use analytic derivatives. Please use Newton's method (or Levenberg-Marquardt if you prefer) to carry out the fit. What are your best-fit parameters for the amplitude, width, and center? Please parameterize the Lorentzian as d = a/(1+(t-t0)**2/w**2)

import numpy as np
from numpy.linalg import inv,pinv


def A(m,t):
    """Evaluates lorentzian model at all times t

    m : tuple or array-like
        Parameters (a, t0, w)
    t : np.ndarray or float
        times at which to evaluate
    """
    a,t0,w=m
    return a/(1+((t-t0)/w)**2)

def A3(m,t): 
    """Evaluates our 3-lorentzians model at all times t

    m : tuple or array-like
        m = (a, b, c, t0, dt, w)
    t : np.ndarray or float
        times at which to evaluate
    """
    a,b,c,t0,dt,w = m # Unpack parameters
    l1 = a/(1 + ((t-t0)/w)**2)    # Compute lorentzians
    l2 = b/(1 + ((t-t0+dt)/w)**2) # Compute lorentzians
    l3 = c/(1 + ((t-t0-dt)/w)**2) # Compute lorentzians
    return l1 + l2 + l3

def gradA(m,t):
    """Analytically compute the gradiant of A wrt m at m,t

    Assumes m is the three vector m=(a,t0,w)
    """
    a,t0,w=m.T
    dAda=1/(1+((t-t0)/w)**2)
    dAdt0=2*a*(t-t0)/(w**2*(1+((t-t0)/w)**2)**2)
    dAdw=2*a*(t-t0)**2/(w**3*(1+((t-t0)/w)**2)**2)
    return np.vstack([dAda,dAdt0,dAdw]).T

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

def numerical_grad(A,m,t):
    """Numerically compute the gradiant of A wrt m at m,t

    A : function
        Assumes A takes two arguments (params, times)
    m : array-like
    t : np.ndarray
    """
    return np.vstack([ndiff(lambda m:A(m,t),m,idx) for idx in range(len(m))]).T
    # Below more legible but doesn't generalize to A3
    # dAda  = ndiff(lambda m:A(m,t),m,0)
    # dAdt0 = ndiff(lambda m:A(m,t),m,1)
    # dAdw  = ndiff(lambda m:A(m,t),m,2)
    # return np.vstack([dAda, dAdt0, dAdw]).T

def newton_iter(m,t,d):
    """Returns next iteration of newton's method"""
    r=d-A(m,t) # residuals
    Ap=gradA(m,t)
    # Ninv=np.identity(len(m)) # Our noise model is pretty simple
    # the Ninv's cancel
    # return m + inv(Ap.T@Ninv@Ap)@Ap.T@Ninv@r
    return m + inv(Ap.T@Ap)@Ap.T@r

def newton_iter_numerical(A,m,t,d):
    """Returns next iteration of newton's method

    A : function / model
        Takes two params (model_params, times)
    m : array-like
        our model parameters, passed to A as first arg
    t : np.ndarray
        Times at which LASER beam is measured
    d : measured data
    """
    r=d-A(m,t) # residuals
    Ap=numerical_grad(A,m,t)
    return m + inv(Ap.T@Ap)@Ap.T@r # Ninv's cancel (do they??)

def mcmc(d,A,m0,t,cov,sigma,nstep,step_size):
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
    ### 1a
    print("INFO: Best fit parames newton iter analytical derivatives")
    dta=np.load("sidebands.npz")
    t=dta['time']
    d=dta['signal']
    print("INFO: Finding best fit parameters")
    # initiate m with some sensible parameters
    w=1.0e-5
    t0=np.mean(t)
    a=np.max(d)
    m0=np.array((a,t0,w))  
    m=m0.copy()
    err=[] # rms residuals with each iteration
    def rmse(r):return np.sqrt(np.mean(r**2))
    err.append(rmse(d-A(m,t)))
    for i in range(10):
        m=newton_iter(m,t,d)
        err.append(rmse(d-A(m,t)))
    print("DEBUG: rmse=")
    for i in err: print(i)
    print("INFO: Chi squared minimized, newton iterations converged.")
    print(f"INFO: The initial parameters where a={a:.3e},t0={t0:.3e},w={w:.3e}")
    a,t0,w=m
    m_analytic=m.copy() # Make a copy of the analytically obtained parameters for
                        # comparison with those obtained w/ numerical derivatives
    print(f"INFO: The best fit parameters are a={a:.3e},t0={t0:.3e},w={w:.3e}")

    ### 1b
    print("\nINFO: Part 1b, approximate noise")
    # approximate the noise, assume uncorrelated
    sigma=np.mean(np.abs(d-A(m,t)))
    #Ninv=np.diag(np.ones(d.shape)/sigma**2)
    Ap=gradA(m,t) # compute gradient once at optimum for further calculations
    print(f"DEBUG: Ap.shape={Ap.shape}, (Ap.T@Ap).shape={(Ap.T@Ap).shape}")
    covar=inv(Ap.T@Ap/sigma**2) # compute covariance matrix linear estimate
    var_a,var_t0,var_w=covar[0,0],covar[1,1],covar[2,2]
    print(f"INFO: Estimate of linearized errors var_a={var_a:.2e}, var_t0={var_t0:.2e}, var_w={var_w:.2e}")
    sig_a,sig_t0,sig_w=np.sqrt(var_a),np.sqrt(var_t0),np.sqrt(var_w)
    print(f"INFO: Estimate of linearized errors sig_a={sig_a:.2e}, sig_t0={sig_t0:.2e}, sig_w={sig_w:.2e}")
    print(f"INFO: Estimate of normalized errors sig_a/a={sig_a/a:.2e}, sig_t0/(tf-ti)={sig_t0/(max(t)-min(t)):.2e}, sig_w/w={sig_w/w:.2e}")


    ### 1c
    print("\nINFO: Part 1 c, using numerical derivatives")
    dta=np.load("sidebands.npz")
    t=dta['time']
    d=dta['signal']
    print("INFO: Finding best fit parameters")
    # initiate m with some sensible parameters
    w=1.0e-5
    t0=np.mean(t)
    a=np.max(d)
    m03=np.array((a,t0,w))  
    m=m03.copy()
    err=[] # rms residuals with each iteration
    def rmse(r):return np.sqrt(np.mean(r**2))
    err.append(rmse(d-A(m,t)))
    for i in range(10):
        m=newton_iter_numerical(A,m,t,d)
        err.append(rmse(d-A(m,t)))
    # print("DEBUG: rmse=")
    # for i in err: print(i)
    print("INFO: Chi squared minimized, newton iterations converged.")
    print(f"INFO: The initial parameters where a={a:.3e},t0={t0:.3e},w={w:.3e}")
    a,t0,w=m
    print(f"INFO: The best fit parameters are a={a:.3e},t0={t0:.3e},w={w:.3e}")
    # Compare with analytic derivatives
    print(f"INFO: Difference between params obtained with analytic vs numerical derivs")
    aa,t0a,wa=m_analytic
    print(f"\tda={a-aa:.2e},\n\tdt0={t0-t0a:.2e},\n\tdw={w-wa:.2e}")
    print("INFO: Compare these with linearized errors, (above should be order of magnitude smaller than below)")
    print(f"\tsig_a={sig_a:.2e},\n\tsig_t0={sig_t0:.2e},\n\tsig_w={sig_w:.2e}")
    m3=m.copy() # Three parameters only

    ### 1d
    print("\nINFO: Part 1d, more complicated model")
    dta=np.load("sidebands.npz")
    t=dta['time']
    d=dta['signal']
    print("INFO: Finding best fit parameters")
    # initiate m with some sensible parameters
    w=1.0e-5
    t0=np.mean(t)
    dt=(t[-1]-t[0])/10
    a=np.max(d)
    b=a/4
    c=a/4
    m06=np.array((a,b,c,t0,dt,w))  
    m=m06.copy()
    err_d=[] # rms residuals with each iteration
    def rmse(r):return np.sqrt(np.mean(r**2))
    err_d.append(rmse(d-A3(m,t)))
    for i in range(10):
        m=newton_iter_numerical(A3,m,t,d)
        err_d.append(rmse(d-A3(m,t)))
    print("DEBUG: rmse=")
    for i in err_d: print(i)
    print("INFO: Chi squared minimized, newton iter converged.")
    print("INFO: The initial parameters where")
    print(f"\ta={a:.2e},\n\tb={b:.2e},\n\tc={c:.2e},\n\tt0={t0:.2e},\n\tdt={dt:.2e},\n\tw={w:.2e}")
    a,b,c,t0,dt,w=m
    print("INFO: The best fit parameters are")
    print(f"\ta={a:.2e},\n\tb={b:.2e},\n\tc={c:.2e},\n\tt0={t0:.2e},\n\tdt={dt:.2e},\n\tw={w:.2e}")
    m6=m.copy() # six parameters vector

    print("\nINFO: Approximate errors")
    # Approximate the noise, assume uncorrelated
    sigma=np.mean(np.abs(d-A3(m6,t)))
    Ap=numerical_grad(A3,m6,t) # compute gradient once at optimum for further calculations
    print(f"DEBUG: Ap.shape={Ap.shape}, (Ap.T@Ap).shape={(Ap.T@Ap).shape}")
    covar=inv(Ap.T@Ap/sigma**2) # compute covariance matrix linear estimate
    sig_a,sig_b,sig_c,sig_t0,sig_dt,sig_w=np.sqrt(np.diag(covar))
    print(f"INFO: Estimate of linearized errors \n\tsig_a={sig_a:.2e}, \n\tsig_b={sig_b:.2e}, \n\tsig_c={sig_c:.2e}, \n\tsig_t0={sig_t0:.2e}, \n\tsig_dt={sig_dt:.2e}, \n\tsig_w={var_w:.2e}")
    print(f"INFO: Estimate of normalized (scaled) errors \n\tsig_a/a={sig_a/a:.2e}, \n\tsig_b/b={sig_b/b:.2e}, \n\tsig_c/c={sig_c/c:.2e}, \n\tsig_t0/(tf-ti)={sig_t0/(max(t)-min(t)):.2e}, \n\tsig_dt/dt={sig_dt/dt:.2e}, \n\tsig_w/w={sig_w/w:.2e}")

    ### Problem 1f
    # The paremeters m6 are the mean, the 
    cov_chol = np.linalg.cholesky(covar)
    n = 5
    dm = np.random.normal(size=(6,n)) # sample from the sample normal
    m_realizations = np.vstack([m for _ in range(n)]).T + cov_chol@dm
    chisq = (d-A3(m6,t)).T@(d-A3(m6,t))/sigma**2
    print(f"INFO: chiquared={chisq:.5e}... and for the pertubations")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    for i in range(n):
        realization = m_realizations[:,i] # perturb around the optimum
        chisq_r = (d-A3(realization,t)).T@(d-A3(realization,t))/sigma**2
        print(f"\t#{i} absolute {chisq_r:.4e}, relative {(chisq_r-chisq)/chisq:.4e}")
        plt.plot(t,A3(m6,t)-A3(realization,t),"-",label=f"diff chisq normalized={(chisq_r-chisq)/chisq:.3e}") # plot the realization
    # plt.plot(t,d,".",label="data")
    #plt.plot(t,d-A3(m6,t),".",label="residuals optimal")
    plt.legend()
    plt.tight_layout()
    plt.title("model pertubation difference between realizations")
    plt.savefig("../img/p1f.png")
    plt.show(block=True)
        
    ### Problem 1g
    print("\nINFO: Running MCMC")
    nstep = 10000 # Number of steps taken in each chain
    step_size = 1.0 # Determines size of step
    # Run chain
    params, chisq = mcmc(d,A3,m6,t,covar,sigma,nstep,step_size)
    mean_params = np.mean(params,axis=1)
    std_params  = np.std(params,axis=1)
    print("INFO: MCMC best fit params are")
    for name,val,std in zip(('a','b','c','t0','dt','w'),mean_params,std_params):
        print(f"\t{name}={val:.4e} +- {std:.1e}")
    # Plot Chi-squared
    plt.figure()
    plt.title("Chi-squared")
    plt.xlabel("Virtual time of markov chain random walker")
    plt.plot(chisq)
    plt.ylabel("chi squared")
    plt.savefig("../img/chi_squared.png")
    plt.show(block=True)
    # Plot mean value of parameters
    plt.figure()
    plt.title("Normalized Parameter means converging like sqrt n")
    param_means = np.array([np.mean(params[:,:idx],axis=1) for idx in range(1,nstep)]).T
    for name,param_mean in zip(('a','b','c','t0','dt','w'),param_means):
        normalized_mean = param_mean / param_mean[-1]
        plt.plot(normalized_mean,label=name)
    plt.legend()
    plt.tight_layout()
    plt.savefig("../img/1g_param_means_converging.png")
    plt.show(block=True)
    # Plot a single parameter
    plt.figure()
    plt.title("Single parameter converging: a")
    a = params[0,:]
    plt.plot(a)
    plt.xlabel("Virtual time")
    plt.ylabel("Value of a")
    plt.tight_layout()
    plt.savefig("../img/1g_param_a.png")
    plt.show(block=True)

    ## Problem 1h
    print("\nINFO: Computing Cavity Width")
    dt = mean_params[4]
    sigma_dt = std_params[4]
    w  = mean_params[5]
    sigma_w = std_params[5]
    cavity_width      = 9*dt/w # in GHz
    uncertainty_width = np.sqrt(sigma_dt**2 + sigma_w**2)/w
    print(f"INFO: Cavity width={cavity_width:.5e}")
    print(f"INFO: Uncertainty in width={uncertainty_width:.2e}")



    # Optionally plot plots by passing v as command line argument
    import sys
    verbose=False
    try: 
        if sys.argv[1]=="v": verbose=True
    except: pass
    if verbose:
        import matplotlib.pyplot as plt
        ### Plots from part 1 (a)
        plt.figure()
        plt.plot(t,d,"x",label="data")
        plt.plot(t,A(m3,t),label="model")
        plt.plot(t,A(m03,t),label="model init guess")
        plt.legend()
        plt.title("Best fit, question1a")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig("../img/q1a_best_fit.png")
        plt.show(block=True)

        plt.figure()
        plt.plot(d-A(m3,t))
        plt.title("Residuals, question1a")
        plt.xlabel("Time")
        plt.ylabel("Amplitude d-A(m)")
        plt.tight_layout()
        plt.savefig("../img/q1a_residuals.png")
        plt.show(block=True)

        plt.figure()
        plt.semilogy(err,'x')
        plt.xlabel("Iteration number")
        plt.ylabel("mean squared residuals")
        plt.title("Newton iter converging fast")
        plt.tight_layout()
        plt.savefig("../img/q1a_newton_converge.png")
        plt.show(block=True)

        # Plots from part 1d
        plt.figure()
        plt.plot(t,d,"x",label="data")
        plt.plot(t,A3(m6,t),label="model")
        plt.plot(t,A3(m06,t),label="model init guess")
        plt.legend()
        plt.title("Best fit, question1d")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig("../img/q1d_best_fit.png")
        plt.show(block=True)

        plt.figure()
        plt.plot(d-A3(m6,t))
        plt.title("Residuals, question1d")
        plt.xlabel("Time")
        plt.ylabel("Amplitude d-A3(m)")
        plt.tight_layout()
        plt.savefig("../img/q1d_residuals.png")
        plt.show(block=True)

        plt.figure()
        plt.semilogy(err_d,'x')
        plt.xlabel("Iteration number")
        plt.ylabel("mean squared residuals")
        plt.title("Newton iter converging fast")
        plt.tight_layout()
        plt.savefig("../img/q1d_newton_converge.png")
        plt.show(block=True)


    




