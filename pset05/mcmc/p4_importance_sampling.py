import json # to save cov and params
import numpy as np
import matplotlib.pyplot as plt

# Load data from chains in problem 3
print("INFO: Loading chain")
data=np.loadtxt("./mcmcdata/plank_chain.txt")
chisq,pars=data[:,0],data[:,1:]
print(f"DEBUG: chisq.shape={chisq.shape}, chain/pars.shap={pars.shape}")

# Determine weights for importance sampling
def get_weights(pars):
    tau=pars[:,3]# sampled taus
    tau_bar=0.0540# prior on tau 
    tau_sig=0.0074# prior sigma on tau
    # Ratio of likelyhoods is just our extra prior
    weights=np.exp(-0.5*((tau-tau_bar)/tau_sig)**2)
    return weights

# Importance sample parameters 
weights=get_weights(pars)
print(f"DEBUG: weights.shape={weights.shape}")
mean_pars=np.average(pars,weights=weights,axis=0)
print(f"DEBUG: mean pars={mean_pars}")

# Construct new importance sampled cov matrix
npar=pars.shape[1]
assert npar==6 # sanity check
cov=np.zeros((npar,npar))#covariance matrix
for i in range(npar):
    for j in range(i+1):
        pars_i=pars[:,i]
        pars_j=pars[:,j]
        mu_i  =mean_pars[i]
        mu_j  =mean_pars[j]
        cov_ij=np.average((pars_i-mu_i)*(pars_j-mu_j),weights=weights)
        cov[i,j]=cov_ij
        cov[j,i]=cov_ij


# Save stuff
parnames=["Hubble constant H0","Baryon density","Dark matter density",
        "Optical depth","Primordial amplitude","Primordial tilt"]
with open("plank_mcmc_importance_params.txt","w") as f:
    json.dump({
        "parnames":parnames,
        "pars":list(mean_pars),
        "cov":[list(i) for i in cov]
        },f,indent=2)


        


