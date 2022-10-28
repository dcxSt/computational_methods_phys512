import numpy as np
import json
from p2_newton import get_spectrum,get_chisq
from matplotlib import pyplot as plt
import datetime



def mcmc(d,A,m0,cov,errs,nstep,step_size):
    """Computes MCMC of chi-squared given our model A(m,t)

    Parameters
    ----------
    d : np.ndarray
        data
    A : function 
        Evaluates the model, takes args (m,t)
    m0 : array-like
        Starting model parameters
    cov : ndarray
        Covariance matrix 
    errs : ndarray
        1d uncorrelated noise in data array. Aka 1/sqrt(Ninv), where 
        Ninv is a 1d array representing a diagonal matrix
    nstep : int
        Number of MCMC steps.
    step_size : float
        Scaling factor, positive float.

    Returns
    -------
    np.ndarray
        A trace of all parameters used along the chain
    np.ndarray
        A trace of all chi-squared values computed along the way
    """
    # Define chi-squared function
    def get_chisq(m):
        model=A(m)
        model=model[:len(d)]
        resid=d-model
        return np.sum((resid/errs)**2)
    # Take Cholesky decomposition to speed things up
    cov_chol = np.linalg.cholesky(cov)
    # Compute chi-squared
    m,chisq = m0.copy(),get_chisq(m0)
    # Initiate data lists
    params_trace = [m] 
    chisq_trace   = [chisq]
    chisq0=get_chisq(M0) # Trace
    # Main loop, wonder around, explore the space
    for idx in range(1,nstep+1):
        print(f"\tStep {idx}/{nstep}")
        print(f"\t\tchisq     ={chisq}")
        print(f"\t\tchisq_diff={chisq-chisq0} should be positive or small negative")
        params_str=" ".join([f"{(i-i0)/i0:.1e}" for i,i0 in zip(m,M0)])
        print(f"\t\tparams diff normalized={params_str}")
        valid_param=False
        # This while loop is so that the program doesn't crash if our 
        # random walker wonders too far. Mathematically, this is
        # equivalent to putting infinite heavy-side at the boundary of
        # where our loss function is valid
        while not valid_param:
            try:
                # Update param
                randvec = np.random.normal(size=m.shape)
                m_next = m + step_size*cov_chol@randvec
                # Compute accept probability
                chisq_next = get_chisq(m_next) # this can throw an error for edge cases
                valid_param=True
            except Exception as e:
                print(f"\nWARNING: \n{e}\n\nWARNING: Re-computing next step")
        delta_chisq = chisq_next - chisq # if next chisq is bigger, it's less likely
        p = np.exp(-0.5*delta_chisq) # if chisq is really big, it'll get small
        if np.random.rand() < p:
            m,chisq = m_next,chisq_next # Update parameters
        # Add step to the parameters list
        params_trace.append(m) 
        chisq_trace.append(chisq)
    return params_trace,chisq_trace

# Get model parameters to init MCMC
dic_in=json.load(open("plank_fit_params.txt","r"))
# Load the last element of the last chain
#pars=np.array(dic_in["pars"])
#m0=pars.copy()

import os
filenames = os.listdir("./mcmcdata")
filenames.sort()
#fname=filenames[-1]
fname=filenames[-3]
print(f"fname=./mcmcdata/{fname}")
m0=np.load("./mcmcdata/"+fname)[-1,:]
pars=m0.copy() 
cov=np.array(dic_in["cov"]) # Covariance matrix

M0=m0.copy() # debug

# Data
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3])

# Run MCMC 
i=0
while True:
    i+=1
    print(f"INFO: Running MCMC, iteration #{i}")
    params_trace,chisq_trace = mcmc(d=spec,
            A=get_spectrum,
            m0=pars,
            cov=cov,
            errs=errs,
            nstep=100,
            step_size=0.50)
    params_trace,chisq_trace=np.array(params_trace),np.array(chisq_trace)
    
    # Save results to hard disk
    print("\nINFO: Saving arrays")
    current_time = datetime.datetime.now().isoformat()
    np.save(f"mcmcdata/{current_time}_params.npy",params_trace)
    np.save(f"mcmcdata/{current_time}_chisq.npy",chisq_trace)
    pars=params_trace[-1,:]

# Print some stuff about the results
print("INFO: MCMC sim metadata")


# Plot results
plt.figure(figsize=(8,5))
for idx in range(params_trace.shape[1]):
    plt.plot(params_trace[:,idx]/m0[idx])
plt.show(block=True)





