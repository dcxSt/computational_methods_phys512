import numpy as np
from matplotlib import pyplot as plt

# shortcuts
sqrt=np.sqrt

# Load MCMC data
import os 
allfiles=os.listdir("mcmcdata")
allfiles.sort()
parsfiles=[i for i in allfiles if "params" in i]
chifiles=[i for i in allfiles if "chisq" in i]
parsdata=[np.load(f"mcmcdata/{i}").T for i in parsfiles]
chidata=[np.load(f"mcmcdata/{i}").T for i in chifiles]
for pars,chi in zip(parsdata,chidata):
    print(f"pars.shape {pars.shape}")
    print(f"chi.shape {chi.shape}")

#pars =np.load(f"mcmcdata/{basename}_params.npy")
#chisq=np.load(f"mcmcdata/{basename}_chisq.npy")
pars=np.hstack(parsdata)
chisq=np.hstack(chidata)
print("COMBINED")
print(f"pars shape {pars.shape}")
print(f"chisq shape {chisq.shape}")

# Trunkate burn in # actually this is unecessary in this case

# Compute and display some key info
parnames=["Hubble constant H0","Baryon density","Dark matter density",
        "Optical depth","Primordial amplitude","Primordial tilt"]
# Estimate the most likely value of each param
mean_pars=np.mean(pars   ,axis=1)
print(f"DEBUG: mean_pars.shape={mean_pars.shape}")
print(f"DEBUG: mean_pars={mean_pars}")
# Estimate the variance in each param
var_pars =np.mean(pars**2,axis=1) - mean_pars**2
# Estimate the covariance matrix
cov = np.diagflat(var_pars) #np.zeros((pars.shape[0],pars.shape[0]))
for i in range(pars.shape[0]):
    for j in range(i):
        cov_ij=np.mean((pars[i,:]-mean_pars[i])*(pars[j,:]-mean_pars[j]))
        cov[i,j]=cov_ij
        cov[j,i]=cov_ij
print(f"Cov mat \n{cov}")

## Plot the mean as a function of time
#means=[np.mean(pars[:i,:],axis=0) for i in range(1,pars.shape[0])] 
#variances=[np.mean(pars[:i,:]**2,axis=0) - np.mean(pars[:i,:],axis=0)**2 for i in range(1,pars.shape[0])]



# Plot Normalized parameters
plt.figure(1,figsize=(9,6))
for par,meanpar,varpar,name in zip(pars,mean_pars,var_pars,parnames):
    print(f"\npar={par}\nmeanpar={meanpar}")
    plt.plot((par-meanpar)/sqrt(varpar),".",markersize=0.5,alpha=0.5,label=name)
plt.legend()
plt.title("Sigma normalized parameters")
plt.xlabel("MCMC time")
plt.ylabel("Parameter value")
plt.savefig(f"mcmcplot/all_params.png",dpi=450)
plt.show(block=False)
plt.pause(0.2)


plt.figure(2,figsize=(9,6))
for par,meanpar,varpar,name in zip(pars,mean_pars,var_pars,parnames):
    plt.loglog(np.abs(np.fft.rfft((par-meanpar)/sqrt(varpar))),"-",linewidth=0.3,alpha=0.5,label=name)
plt.title("Abs RFFT sigma normalized params")
plt.legend()
plt.xlabel("MCMC freq-space")
plt.ylabel("Parameter's Random-Walk fourier eigenvalue")
plt.savefig(f"mcmcplot/all_params_fft.png",dpi=450)
plt.show(block=True)




