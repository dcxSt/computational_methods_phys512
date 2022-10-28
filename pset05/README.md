*We will use the power spectrum of the Cosmic Microwave Background (CMB) to constrain the basic cosmological parameters of the universe. The parameters we will measure are the Hubble constant, the density of regular baryonic matter, the density of dark matter, the amplitude and tilt of the initial power spectrum of fluctuations set in the very early universe, and the Thompson scattering optical depth between us and the CMB. In this excercise, we will only use intensity data, which does a poor job constraining the optical depth.*

*For the data, we will use data from the Planck satellite, which I have helpfully downloaded for you. Look for `COM_PowerSpect_CMB-TT-full_R3.01.txt` in the mcmc directory. This gives the variance of the sky as a function of angular scale, and the uncertainty of the variance. The columns are 1) multipole (starting with l=2 quadrupole), 2) the variance of the sky at that multipole 3) the 1-sigma lower uncertainty, and 4) the 1-sigma upper uncertainty. To make your lives easy, assume the errors are Gaussian and uncorrelated, and that the error on each point is the average of the upper and lower errors. This is one of several simplifications we'll make so our answers won't be exactly correct, but they will be very close.*

We load the data and define our errors
```python
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3])
```

*You'll also need to be able to calculate model power spectra as a function of input parameters. You can get the source code for CAMB from [Antony Lewis's github page](github.com/cmbant). There's a short tutorial [on line](camb.readthedocs.io/in/latest/CAMBdemo.html) as well. Note that CAMB returns the power spectrum starting with the monopole, so you may need to manually remove the first two entries.*

This function helps us compute the spectrum

```python
def get_spectrum(pars,lmax=3000):
  H0=pars[0]
  ombh2=pars[1]
  omch2=pars[2]
  tau=pars[3]
  As=pars[4]
  ns=pars[5]
  pars=camb.CAMBparams()
  pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
  pars.InitPower.set_params(As=As,ns=ns,r=0)
  pars.set_for_lmax(lmax,lens_potential_accuracy=0)
  results=camb.get_results(pars)
  powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
  cmb=powers['total']
  tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
  return tt[2:] # clip off monopole and dipole
```

## 1)
*Based on the chi-squared value, are the parameters dialed into my test script an acceptable fit?*

The parameters are `[60,0.02,0.1,0.05,2.0e-9,1.0]`. These are not acceptable. First of all, the fit doesn't go through 60% of the error bars. Second, for these parameters, chi-squared evaluates to `15268` and change. As we will see, there are better chi-squared values to be found in this parameter space. 

*What do you get for chi-squared for parameters equal to `[69, 0.022, 0.12, 0.06, 2.1e-9, 0.95]`, which are closer to their current-accepted values?*

The chi-squared value for these parameters is `3272.2036739044693`. This is an improvement of about `12000`. Since chi-squared is a log likelyhood, this means these parameters are e^12000 times more likely?

*Would you consider these values an acceptable fit? Note - the mean and variance of chi-squared are n and 2n, respectively, where n is the number of degrees of freedom.*

# TODO!!

*2) Use Newton's method or Levenberg-Marquardt to find the best-fit parameters, using numerical derivatives. Your code should report your best fit parameters and their errors in `plank_fit_params.txt`. Please write your own fitter/numerical-derivative-taker rather than stealing one. Note - you will want to keep track of the curvature matrix at the best-fit value for th enext problem.*

To do this fit, we need to take gradients, for which we need to take partial derivatives. We do so using the following code. 

```python
### Numerical derivatives
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
    # (ballpark back of envolope estimate, assume f order 1)
    dx = 2.0e-08
    step_h = np.zeros(x.shape)
    step_h[idx] = dx
    # Return the numerical partial derivative of f wrt it's argument at 
    # idx, at x
    return (f(x+step_h) - f(x-step_h))/(2*dx)

def numerical_grad(f,x):
    """Numerically compute the gradiant of f
    
    f : function (think, chi-squared)
    x : parameters (think, model params)

    returns : ndarray
        dims  len(f(x)) X len(x) 
    """
    return np.vstack([ndiff(f,x,idx) for idx in range(len(x))]).T
```

Now we put this together into a newton iterator. 

```python
def newton_iter(A,m,d,m0):
    """Iterate newton's method to minimize (A(m)-spec).T@Ninv@(A(m)-spec)

    We don't need the errors because, for our purposes they are 
    diagonal. 

    Parameters
    ----------
    A : function
        The model, takes the vector argument m
    m : ndarray
        Model parameters
    d : ndarray
        The data we are fitting to. Dimensions must match size 
        of output of A(m).
    m0 : ndarray
        To make sure our parameters are more or less optimal scale for 
        the numerical derivatives. The size must match the size of m. 

    Returns
    -------
        Next iteration of best-guess of optimal parameters. 
    """
    # This scaled model function uses it's environment, 
    # like a closure in other languages
    def A_scaled(m_scaled):
        model=A(m_scaled*m0) 
        model=model[:len(d)] # trunkate model to fit data
        return model
    m_scaled=m/m0               # Scale the input appropriately
    model=A_scaled(m_scaled)    # Evaluate the model
    resid=spec-model            # Compute residuals
    grad_scaled=numerical_grad(A_scaled,m_scaled) # Scaling important for this step
    m_scaled += pinv(grad_scaled.T@grad_scaled)@grad_scaled.T@resid # Update
    return m_scaled*m0          # Re-scale and return
```

And use it to find the minimum of chi-squared. 

```python
for i in range(11):
  pars=newton_iter(get_spectrum,pars,spec,m0)
```

Unfortunately, our value wonders around the optimum a bit. This is probably because the function we are optimizing is not well behaved (smooth, parabolic) near the minimum. 

*Bonus: The CMB is some of the best evidence we have for dark matter. What are the best-fit parameters with the dark-matter density set to zero? How does chi-squared compare tot he standard value? Note - getting this to converge can be tricky, so you might want to slowly step down the dark matter density to avoid crashes. If you get this to work, print the parameters/errors in `plank_fit_params_nodm.txt`*

See `p2_bonus.py`

The best fit params for no-dark-matter can be found in `plank_fit_params_nodm.txt`. The chi-squared for this is 10648.389, so these model params are about e^8000 x less likely than with optimal dark-matter. We obtained this fit by slowly varying the dark matter parameter and fitting each time. More precisely, here is some pythonic pseudocode:

```
initiate all variables to the optimal values calculated in problem 2
for each dark-matter value in linspace(optimal dark matter, zero, 20):
  do three newton iterations to optimize chi-sq in all the other parameters
once you reach dark-matter density = zero
iterate newton's method until converged
```

## 3) 
*Estimate the parameter values and uncertainties using an MCMC sampler you write yourself. I strongly suggest you draw your trial steps from the curvature matrix you generated in Problem 2. Save your chain (includig the chi-squared value for each smaple in the first column) in `plank_chain.txt`*

Code for this problem is in `p3_mcmc.py`. Then I use the script `/mcmcdata/npy_to_txt.py` to put my data into the format asked for (in `plank_chain.txt`) (although I don't want to commit this because it's a 28M file...), and also to get the best fit params, which I will definitely commit, and can be found in json format in `plank_mcmc_fit_params.txt`. 

Here is the mcmc algorithm

```python
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
        Covariance matrix. Determines distribution of random step. 
    errs : ndarray
        1d uncorrelated noise in data array. Aka 1/sqrt(Ninv), where 
        Ninv is a 1d array representing a diagonal matrix
    nstep : int
        Number of MCMC steps.
    step_size : float
        Scaling factor, positive float. Determines scale of random step

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
```

We get these parameters, which are in `plank_mcmc_fit_params.txt` (not displayed here is also the cov matrix). To get the best fit params, we trunkate the burn in and keep the long tail—so we throw out the first 30000 steps (we choose 30000 as a reasonable estimate from our plot). 

```
{
  "chisq": 2582.969437409757,
  "parnames": [
    "Hubble constant H0",
    "Baryon density",
    "Dark matter density",
    "Optical depth",
    "Primordial amplitude",
    "Primordial tilt"
  ],
  "pars": [
    70.97876519206841,
    0.022687505915976715,
    0.11163315049501213,
    0.11908555436918732,
    2.34152678118161e-09,
    0.9869602906873022
  ]
}
```

We can tell our chain has converged by observing that the lower frequencies look like white noise. 

The dark matter density times h-squared is `0.1116 +- 9.5e-4`. To get the actual dark-matter density we divide by h-squared, which is H0/100. `H0 = 71.0 +- 0.25`, so `h = 0.710 +- 0.0025`. We can approximate the square and std of the square by `sigma_hsq = 2*sigma_h*h` to first order, `h^2 = 0.504 +- 0.0035`. Now, to get the dark-matter density we divide our original number by `h^2`, keeping only highest order terms to estimate our uncertainty 

```
Omega \approx \Omega_hsq/hsq \pm (sigma_omega_hsq + sigma_hsq)/hsq
Omega = 0.1116/0.504 +- (0.0035 + 9.5e-4)/0.504
      = 0.221 +- 0.0088
```



![all_params_fft](https://user-images.githubusercontent.com/21654151/198725023-5c5db8e9-d13e-4979-aafe-3a55cbd2da68.png)

![all_params](https://user-images.githubusercontent.com/21654151/198724988-31fad741-ae70-4eb7-ae71-f3fdfd7e2470.png)


![covmat_mcmc](https://user-images.githubusercontent.com/21654151/198724967-5a4ed4e0-933f-4476-81ea-4c800407c3a8.png)


## 4)

**Importance Sampling**

We use importance sampling by weighting the averages with the ratio of likelyhoods of our original loss function with the one that includes the new prior on tau. The ratio is just the exponential of `-0.5` times our prior componant of our new chi-squared function, which we assume to be a quadratic, as it's a good assumption to make that our tau likelyhood, obtained from polarization data, is normally distributed. This is done in the code snippet below, which can be found in `p4_importance_sampling.py`. We use importance sampling  `plank_mcmc_importance_params.txt`. 

```python
# Determine weights for importance sampling
def get_weights(pars):
    tau=pars[:,3] # sampled taus
    tau_bar=0.0540 # prior on tau 
    tau_sig=0.0074 # prior sigma on tau
    # Ratio of likelyhoods is just our extra prior
    weights=np.exp(-0.5*((tau-tau_bar)/tau_sig)**2)
    return weights
```

The covariance of two parameters can be estimated like so

```python
pars_i=pars[:,i]
pars_j=pars[:,j]
mu_i=np.average(pars_i,weights)
mu_j=np.average(pars_j,weights)
cov_ij=np.average((pars_i-mu_i)*(pars_j-mu_j),weights=weights)
```

After importance sampling, our best fit parameters (saved to `plank_mcmc_importance.txt`) are as follows:

```
{
  "parnames": [
    "Hubble constant H0",
    "Baryon density",
    "Dark matter density",
    "Optical depth",
    "Primordial amplitude",
    "Primordial tilt"
  ],
  "pars": [
    70.62405203154053,
    0.022636048025946476,
    0.1123185875192795,
    0.05784293762241056,
    2.0747648182055052e-09,
    0.9840587922443713
  ],
  "vars": [
    0.4143958180902279,
    3.953675332107864e-08,
    1.5217522491607208e-06,
    5.244406228349512e-05,
    9.34928436145934e-22,
    1.5791032739638025e-05
  ]
}
```

Our new normalized covariance matrix looks like this






**New Chain**

To constrain tau, we need only modify our chi-squared function as such

```python
def get_chisq(m):
    """Takes model param, uses environment for A and d"""
    model=A(m)
    model=model[:len(d)]
    resid=d-model
    # Constrain Tau
    tau=m[3]
    taubar=0.0540 # prior for tau
    tausig=0.0074 # prior sigma for tau
    chisq_tau=((tau-taubar)/tausig)**2
    return np.sum((resid/errs)**2) + chisq_tau
```

This chain converged faster. We compute the expected values for the parameters and the new covariance matrix. 











