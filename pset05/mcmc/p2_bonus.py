from p2_newton import *
import json

def newton_iter_subset(A,m,d,m0,Ninv,idx_select):
    """Do newton iter on subset of params

    idx_select selects indices of m we are interested in
    """
    def A_scaled(m_scaled):
        model=A(m_scaled*m0)
        model=model[:len(d)]
        return model
    m_scaled=m/m0
    model=A_scaled(m_scaled)
    resid=d-model
    grad_scaled=numerical_grad(A_scaled,m_scaled)
    grad=grad_scaled/m0
    grad_select=grad[:,idx_select]
    cov=inv((grad_select.T*Ninv)@grad_select)
    m_new=m.copy()
    m_new[idx_select] += cov@(grad_select.T*Ninv)@resid
    return m_new


# Get model parameters optimal model parameters
dic_in=json.load(open("plank_fit_params.txt","r"))
pars=dic_in["pars"]
pars=np.array(pars)
m0=pars.copy()
dark_matter_density0=m0[2]

# Data
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3])
Ninv=1/errs**2 # Inverse diagonal matrix of errors

# Progressively tone down the dark matter,
# aim to tone down small enough that you can 1-shot newton
niter=20
for i,dark_matter_density in enumerate(np.linspace(dark_matter_density0,0,niter)):
    print(f"\nIteration #{i+1}/{niter}")
    print(f"\tdark matter density set to {dark_matter_density}")
    pars[2]=dark_matter_density
    print(f"Chisq={get_chisq(pars,spec,errs,get_spectrum)}")
    idx_select=np.array([0,1,3,4,5])
    pars=newton_iter_subset(get_spectrum,pars,spec,m0,Ninv,idx_select)
    print(f"NI #1 Chisq={get_chisq(pars,spec,errs,get_spectrum)}")
    pars=newton_iter_subset(get_spectrum,pars,spec,m0,Ninv,idx_select)
    print(f"NI #2 Chisq={get_chisq(pars,spec,errs,get_spectrum)}")
    pars=newton_iter_subset(get_spectrum,pars,spec,m0,Ninv,idx_select)
    print(f"NI #3 Chisq={get_chisq(pars,spec,errs,get_spectrum)}")

print("INFO: Newton iters to converge onto right value")
for i in range(15):
    print(f"#{i+1}/15",end=", ")
    pars=newton_iter_subset(get_spectrum,pars,spec,m0,Ninv,idx_select)
    print(f"\tChisq={get_chisq(pars,spec,errs,get_spectrum)}")

# Get the covariance and sigma matrix
cov=get_covariance_matrix(get_spectrum,pars,spec,m0,Ninv)
sigma_pars=np.sqrt(np.diag(cov))
parnames=dic_in["parnames"]

# Serializing
print("\nINFO: Serializing to plank_fit_params_nodm.txt")
dic_out={
    "chisq":get_chisq(pars,spec,errs,get_spectrum),
    "parnames":parnames,
    "pars":list(pars),
    "sigma_pars":list(sigma_pars),
    "cov":[[i for i in j] for j in cov]
        }
with open("plank_fit_params_nodm.txt","w") as f:
    json.dump(dic_out,f,indent=2)
print("\nDone")


    



# Do a couple 
