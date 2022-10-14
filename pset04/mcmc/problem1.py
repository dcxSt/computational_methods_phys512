# To start, model the data as a single Lorenztian and use analytic derivatives. Please use Newton's method (or Levenberg-Marquardt if you prefer) to carry out the fit. What are your best-fit parameters for the amplitude, width, and center? Please parameterize the Lorentzian as d = a/(1+(t-t0)**2/w**2)

import numpy as np
from numpy.linalg import inv,pinv


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
    dAdw=2*a*(t-t0)**2/(w**3*(1+((t-t0)/w)**2)**2)
    return np.vstack([dAda,dAdt0,dAdw]).T

def newton_iter(m,t,d):
    """Returns next iteration of newton's method"""
    r=d-A(m,t) # residuals
    Ap=gradA(m,t)
    # Ninv=np.identity(len(m)) # Our noise model is pretty simple
    # return m + inv(Ap.T@Ninv@Ap)@Ap.T@Ninv@r
    return m + inv(Ap.T@Ap)@Ap.T@r


if __name__=="__main__":
    ### 1a
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
    print("INFO: Chi squared minimized, gradient descent converged.")
    print(f"INFO: The initial parameters where a={a:.3e},t0={t0:.3e},w={w:.3e}")
    a,t0,w=m
    print(f"INFO: The best fit parameters are a={a:.3e},t0={t0:.3e},w={w:.3e}")

    ### 1b
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


    

    import matplotlib.pyplot as plt
    ### Plots from part 1 (a)
    plt.figure()
    plt.plot(t,d,"x",label="data")
    plt.plot(t,A(m,t),label="model")
    plt.plot(t,A(m0,t),label="model init guess")
    plt.legend()
    plt.title("Best fit, question1a")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("../img/q1a_best_fit.png")
    plt.show(block=True)

    plt.figure()
    plt.plot(d-A(m,t))
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


    



