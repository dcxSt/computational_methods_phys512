# Pseudo Random Number Generators

## 1)
*Code can be found in* `PRNG.ipynb`

We make a 3d plot of the c-lib's generated random numbers and rotate it until the stacked-planes pattern emerges

<img width="481" alt="p1_dots" src="https://user-images.githubusercontent.com/21654151/202530204-854430b9-8824-404f-907c-57a5371335e7.png">

We can also see some slightly suspect spikes in the PSD of an estimate of the density function

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




