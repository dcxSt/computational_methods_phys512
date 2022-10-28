import numpy as np
import json
from matplotlib import pyplot as plt

dic=json.load(open("plank_mcmc_fit_params.txt","r"))
cov=np.array(dic["cov"])
mean_pars=dic["pars"]
parnames=dic["parnames"]

# Normalize the covariance matrix
dims=cov.shape
assert dims[0]==dims[1], "Error, cov mat not square!"
norm_cov=np.identity(dims[0]) # diagonals of normaliized cov mat are 1
print("DEBUG: ")
for i in range(dims[0]):
    for j in range(i):
        sigma_i=np.sqrt(cov[i,i])
        sigma_j=np.sqrt(cov[j,j])
        norm_cov[i,j]=cov[i,j]/(sigma_i*sigma_j)
        norm_cov[j,i]=cov[i,j]/(sigma_i*sigma_j)
        print(f"\t{cov[i,j]/(sigma_i*sigma_j):.2f}",end=", ")
    print()


# Plot normalized cov matrix
plt.figure(figsize=(7,7))
plt.title("Normalized Covariance Matrix")
plt.imshow(norm_cov,cmap="RdBu",vmin=-1,vmax=1)
plt.colorbar()
plt.xticks(ticks=np.arange(6),labels=parnames,rotation=65)
plt.yticks(ticks=np.arange(6),labels=parnames)
plt.tight_layout()
plt.savefig("./mcmcplot/covmat_mcmc.png",dpi=450)
plt.show(block=True)


