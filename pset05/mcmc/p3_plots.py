import numpy as np
from matplotlib import pyplot as plt

# shortcuts
sqrt=np.sqrt

# Load a chain 
basename="2022-10-19T21:26:20.889518"
pars =np.load(f"mcmcdata/{basename}_params.npy")
chisq=np.load(f"mcmcdata/{basename}_chisq.npy")

# Trunkate foreburn (???what the name???)

# Compute and display some key info
parnames=["Hubble constant H0","Baryon density","Dark matter density",
        "Optical depth","Primordial amplitude","Primordial tilt"]
mean_pars=np.mean(pars,axis=0)
var_pars =np.mean(pars**2,axis=0) - np.mean(pars,axis=0)**2

# Plot the mean as a function of time
means=[np.mean(pars[:i,:],axis=0) for i in range(1,pars.shape[0])] 
variances=[np.mean(pars[:i,:]**2,axis=0) - np.mean(pars[:i,:],axis=0)**2 for i in range(1,pars.shape[0])]

# Plot Normalized parameters

plt.figure(figsize=(9,6))
for par,meanpar,varpar,name in zip(pars.T,mean_pars,var_pars,parnames):
    plt.plot((par-meanpar)/sqrt(varpar),label=name)
plt.legend()
plt.title("Sigma normalized parameters")
plt.xlabel("MCMC time")
plt.ylabel("Parameter value")
plt.savefig(f"mcmcplot/{basename}_params.png",dpi=450)
plt.show(block=True)




