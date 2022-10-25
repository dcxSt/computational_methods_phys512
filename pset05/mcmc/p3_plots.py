import numpy as np
from matplotlib import pyplot as plt

# shortcuts
sqrt=np.sqrt

# Load a chain 
#basename="2022-10-19T21:26:20.889518"
#basename="2022-10-19T22:21:11.430998"
#basename="2022-10-20T17:22:23.375177"
#basename="2022-10-20T20:16:54.072785"
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
parsdata=parsdata[4:]
chidata=chidata[4:]

#pars =np.load(f"mcmcdata/{basename}_params.npy")
#chisq=np.load(f"mcmcdata/{basename}_chisq.npy")
pars=np.hstack(parsdata)
chisq=np.hstack(chidata)
print("COMBINED")
print(f"pars shape {pars.shape}")
print(f"chisq shape {chisq.shape}")

# Trunkate foreburn (???what the name???)

# Compute and display some key info
parnames=["Hubble constant H0","Baryon density","Dark matter density",
        "Optical depth","Primordial amplitude","Primordial tilt"]
mean_pars=np.mean(pars,axis=0)
var_pars =np.mean(pars**2,axis=0) - np.mean(pars,axis=0)**2
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

#plt.figure(figsize=(9,6))
for par,meanpar,varpar,name in zip(pars,mean_pars,var_pars,parnames):
    plt.figure(figsize=(9,6))
    plt.plot(par,label=name)
    plt.legend()
    plt.show(block=True)
    #plt.plot((par-meanpar)/sqrt(varpar),label=name)
plt.legend()
plt.title("Sigma normalized parameters")
plt.xlabel("MCMC time")
plt.ylabel("Parameter value")
plt.savefig(f"mcmcplot/all_params.png",dpi=450)
plt.show(block=True)




