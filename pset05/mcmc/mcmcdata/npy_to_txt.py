# Take .npy data
#   1. combine and save them as txt single file `plank_chain.txt`
#   2. mean params and their covariances, save that as `plank_mcmc_fit_params.txt` in json human-readable format


import numpy as np
import sys
sys.path.append("..")
from p2_newton import get_spectrum,get_chisq
import json

# shortcuts
sqrt=np.sqrt

# Load MCMC data
import os 
allfiles=os.listdir("./")
allfiles.sort()
parsfiles=[i for i in allfiles if "params" in i]
chifiles=[i for i in allfiles if "chisq" in i]
parsdata=[np.load(f"./{i}").T for i in parsfiles]
chidata=[np.load(f"./{i}").T for i in chifiles]
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


# Load planck model related data 
parnames=["Hubble constant H0","Baryon density","Dark matter density",
        "Optical depth","Primordial amplitude","Primordial tilt"]
planck=np.loadtxt('../COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3])
Ninv=1/errs**2
print("INFO: dumping estimated params")
with open("../plank_mcmc_fit_params.txt","w") as f:
    json.dump({
        "chisq":get_chisq(mean_pars,spec,errs,A=get_spectrum),
        "parnames":parnames,
        "pars":list(mean_pars),
        "cov":[list(i) for i in cov]
        },f,indent=2)

print("INFO: dumping data arrays into txt")

print(f"pars.shape={pars.shape}, chisq.shape={chisq.shape}")
np.savetxt("plank_chain.txt",np.vstack([chisq,pars]))






