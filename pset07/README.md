# Pseudo Random Number Generators

## 1)
*Code can be found in* `PRNG.ipynb`

We make a 3d plot of the c-lib's generated random numbers and rotate it until the stacked-planes pattern emerges

<img width="481" alt="p1_dots" src="https://user-images.githubusercontent.com/21654151/202530204-854430b9-8824-404f-907c-57a5371335e7.png">

We can also see some slightly suspect spikes in the PSD of an estimate of the density function. However, there should be modes at higher frequencies... Maybe it's aliasing? Not enough data here, but thought I'd stick it in for interest reasons. Need to optimize code and get a better estimate. 

```python
# Estimate density function in this triple for loop (NB: unoptimized)
# `rp` is the random points variable
def get_rhoxyz(rp,x,y,z,dx,dy,dz):
    idx0=np.where(rp[:,0]>=x-dx/2)
    idx1=np.where(rp[:,0]< x+dx/2)
    idx2=np.where(rp[:,1]>=y-dy/2)
    idx3=np.where(rp[:,1]< y+dy/2)
    idx4=np.where(rp[:,2]>=z-dz/2)
    idx5=np.where(rp[:,2]< z+dz/2)
    idxs=np.intersect1d(idx0,idx1)
    idxs=np.intersect1d(idxs,idx2)
    idxs=np.intersect1d(idxs,idx3)
    idxs=np.intersect1d(idxs,idx4)
    idxs=np.intersect1d(idxs,idx5)
    return len(rp[idxs])
n=len(rp)
nx,ny,nz = int(n**(1/3)),int(n**(1/3)),int(n**(1/3))
lx,ly,lz = max(rp[:,0])-min(rp[:,0]),max(rp[:,1])-min(rp[:,1]),max(rp[:,2])-min(rp[:,2])
dx,dy,dz = lx/nx,ly/ny,lz/nz
x=np.linspace(min(rp[:,0]),max(rp[:,0]),nx)
y=np.linspace(min(rp[:,1]),max(rp[:,1]),ny)
z=np.linspace(min(rp[:,2]),max(rp[:,2]),nz)
rho=np.zeros((nx,ny,nz))
print("nx=",nx)
for ix,i in enumerate(x):
    print("iter=",ix)
    for jx,j in enumerate(y):
        for kx,k in enumerate(z):
            rho[ix,jx,kx]=get_rhoxyz(rp,i,j,k,dx,dy,dz)
```

![rho_spec](https://user-images.githubusercontent.com/21654151/202530299-f272e7c8-c6d2-4679-a70c-50e0081b6dbe.png)


Python's random number generator does not exhibit this behaviour. (though this is hardly a rigorous proof that the pseudo random number generator exhibits the pseudo random properties we would like it to). 

```python
n=300000000
vec=np.random.rand(3*n)
rp=vec.reshape((n,3))
truncate=10000
x,y,z=rp[:truncate,0],rp[:truncate,1],rp[:truncate,2]
# plot xyz and rotate, see that it's homogenously uniform random
```

<img width="325" alt="p1_python_noplane" src="https://user-images.githubusercontent.com/21654151/202530399-192b4fca-91df-4b89-85a4-3db0d4c35644.png">


We run this c/python snippet that wraps our favourite faulty c-PRNG. It works, and the image is the same as the one above. (see `rand_points_steve.txt`)

```python
import numpy as np
import ctypes
import numba as nb
import time
import matplotlib.pyplot as plt

mylib=ctypes.cdll.LoadLibrary("libc.dylib")
rand=mylib.rand
rand.argtypes=[]
rand.restype=ctypes.c_int

@nb.njit # get numba to pre-compile & accelerate for loop (this is so cool!)
def get_rands_nb(vals):
  n=len(vals)
  for i in range(n):
    vals[i]=rand()
  return vals

def get_rands(n):
  vec=np.empty(n,dtype='int32')
  get_rands_nb(vec)
  return vec
```

## 2)

*Code in* `p2.py`

*Write a rejection method to generate exponential deviates from another distribution. Which of Lorentzians, Gaussians, and power laws could you use for the bounding distribution.*

Asymtotically lorentzians go like $1/x^2$, so they can bound exponentials from above. Gaussians go like $\exp(-x^2)$, so asymptotically $\lim_{x\to\infty}e^{-x}/e^{-x^2}\to e^x$, so the gaussian will eventually dip below the exponential, so we can't use a gaussian. We could use any power law too, because exponentials die faster than polynomials. 

So lets use a power law. We can sample from our power law by inverting the CDF. If the PDF is proportional to $s^{-\alpha}$, where $s$ ranges from 1 to infinity, then the (properly normalized!) CDF is $s^{1-\alpha}$. 

```python
def sample_power_law(n,alpha):
   # Samples from s^-alpha
   y=np.random.rand(n)
   # power law transformation
   x=y**(1/(1-alpha))# inverse CDF
   return x
```

![p2_plaw](https://user-images.githubusercontent.com/21654151/202577614-363bfe0c-ea46-436a-be39-cf63bec9928d.png)


We can use $\alpha=2$ to bound $e^{-x}$. 

![p2_powerlaw_bound_exp](https://user-images.githubusercontent.com/21654151/202577683-d36d8665-85ad-48c6-a02b-371b8fc9c6a9.png)

Now we implement a rejection algorithm. Lets trunkate at $s=10$. 

```python
def sample_reject(g,f,sample_f,n:int):
    """
    Sample n from f's dist (sample_f). Compute random height,
    accept those that land under the g curve.
    """
    # sample f 
    f_samp=sample_f(n)
    # reject some of these
    y_samp=np.random.rand(len(f_samp))*f(f_samp)
    g_select=f_samp[np.where(y_samp<=g(f_samp))]
    g_reject=f_samp[np.where(y_samp> g(f_samp))]
    return g_select,g_reject,y_samp
```

Here is how the sampleing is done. 

![p2_accepted_rejected](https://user-images.githubusercontent.com/21654151/202577723-6c093d3a-4d0e-4bf5-baaf-7ef89d409055.png)

![p2_hist](https://user-images.githubusercontent.com/21654151/202577738-49e37b2c-ebfb-41c9-88b1-e6b71060e26c.png)

There are two ways we can make that blue region (rejects) skinnier. We can change $\alpha$ or we can divide by some factor $\beta$. In math, this reads

$$
\inf_{\alpha,\beta} \int_1^\infty s^{-\alpha}/\beta - e^{-s} ds \quad\bigg|\quad
s^{-\alpha}/\beta \geq e^{-s}\,\,\forall s\in [1,\infty)
$$

This problem is non-trivial. So we do some trial and error and get it to look skinnier. Here's one that doesn't look so bad:


![p2_accepted_rejected_skinny](https://user-images.githubusercontent.com/21654151/202577789-0d5e98ce-19bf-4e90-b0bb-dab941ed3a48.png)


## 3)
*Ratio of uniforms sampler. See code in* `p3.py`

Solving for $v$, we see that the $u$ must satisfy

$$
0<u<\sqrt(p(v/u)) \Rightarrow v<-2u\ln(u) \quad u\in (0,1)
$$


(Analytically computed bounding region)

![p3_rou_acceptance_region](https://user-images.githubusercontent.com/21654151/202617964-f1fcb656-1e22-4a9e-b7ce-b73cffeb379d.png)

There is only one optimum in $u\in(0,1)$ and that's the maximum. We can find it by taking a derivative.

$$
v_{\text{sup}}(u)=-2u\ln u \rightarrow \frac{dv}{du} = -2\ln u - 2 =0
$$

solving for u and v we get

$$
\ln u = -1 \Rightarrow u=e^{-1}\quad v_{\text{sup}}(1/e) = 2/e
$$

So the area is bounded by the box $(x,y) : x\in[0,1], y\in[0,2/e]$. To use this to sample from 1/x, we generate random points uniformly within this bounding box, and evaluate those that fall in the refion at u/v.

```python
# The height of the max accepted v
def droplet(u):
    return -2*u*np.log(u)

# Use ROU to sample from an exponential distribution
def sample_exp(n:int):
    ymax=2/np.exp(1) # Height of accept region
    x=np.random.rand(n)      # Select random points in
    y=np.random.rand(n)*ymax # bounding box
    idxs=np.where(y<droplet(x)) # Accepted indices
    return y[idxs]/x[idxs] # return accepted values
```


For one set of samples, the accepted/rejected values of $u$ and $v$ will look like

![p3_sample_droplet](https://user-images.githubusercontent.com/21654151/202618048-d610d712-02f1-4a83-8006-c195fb8604ed.png)

Indeed our samples generate the correct distribution, as can be seenin this plot of the exponential. 
![p2_plaw](https://user-images.githubusercontent.com/21654151/202618174-9e684f42-8024-49d9-b52a-f64f96ff8bea.png)


