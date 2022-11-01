
## 1)

```python
def dftshift(arr:np.ndarray, x0:int):
    """shifts arr by x0, f(x)-->f(x-x0)
    Only works for even lengthed arrays, to make use of irfft"""
    assert len(arr)%2==0, f"array length {arr.shape} must be even 1-d"
    N=len(arr)
    arrft=rfft(arr)
    twidle=np.exp(-2.0j*np.pi*x0/N*np.arange(0,N//2+1))
    shifted=irfft(arrft*twidle)
    return shifted
```

TODO: plot


## 2)

**(a)** The correlation of a guassian with it's self is just a gaussian. This is to be expected because we know that the convolution of two gaussians is a gaussian, and that gaussians symetric about zero are even functions, so we expect the correlation to be equal to the convolution. When they are not centered but shifted by some amount, the correlation is just the convolution of the same shifted by that amount in the opposite direction. 

**(b)** In the case of a gaussian, a shift in one of the inputs will result in a shift in the output of the correlation function. 

```python
def correlation(u,v):
    """Returns correlation of u and v"""
    return irfft(rfft(u)*np.conj(rfft(v)))

def autocorr_shifted(u, shift):
    """correaltion of a function with a shifted version of it's self"""
    u_shifted=np.roll(u,shift)
    return correlation(u,u_shifted)
```

TODO: put plot here

## 3)

To avoid circulant boundary conditions, we can pad each of the input arrays with zeros. We need as many zeros as the length of the arrays for there not to be any wrap-around. 

We can do this by padding then re-cycling our correlation function
```python
def correlation_pad(u,v):
    """Returns correlation of u and v without any wrap-around"""
    zeros=np.zeros(len(u)) # assume len(u)==len(v)
    upad,vpad=np.hstack((u,zeros)),np.hstack((v,zeros))
    return correlation(upad,vpad)
```

## 4) 



