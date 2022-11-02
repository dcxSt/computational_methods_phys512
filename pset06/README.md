
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

**(a)** Show that

$$
\sum_{x=0}^{N-1} \exp(-2\pi ikx/N) = \frac{1 - \exp(-2\pi i k)}{1 - \exp(-2\pi ik/N)}
$$

To see this, we multiply the left hand side by the denominator of the right

$$
(1 - e^{-2\pi ik/N})\sum_{x=0}^{N-1}e^{-2\pi ikx/N} 
= \sum_{x=0}^{N-1} e^{-2\pi ikx/N} - \sum_{x=0}^{N-1} e^{-2\pi ik(x+1)/N}
$$

Observe that terms from the left sum match with terms from of the right sum as we iterate through x. Only the first term from the left sum, which is $e^{-2\pi ik\cdot 0/N}=1$ and the final term from the right sum do not cancel $e^{-2\pi ikN/N} = e^{-2\pi ik}$

Therefore

$$
(1-e^{-2\pi ik/N})\cdot \sum_{x=0}^{N-1}e^{-2\pi ikx/N} = 1 - e^{-2\pi i k}
$$


**(b)**

As $k\to 0$ we have

$$
\lim_{k\to 0}\frac{1 - \exp(-2\pi ik)}{1 - \exp(-2\pi ik/N)} = 
\lim_{k\to 0}\frac{\frac{d}{dk}(1 - \exp(-2\pi ik))}{\frac{d}{dk}(1 - \exp(-2\pi ik/N))} = 
\lim_{k\to 0}\frac{2\pi i \exp^{-2\pi ik}}{2\pi i/N \exp(-2\pi i k/N)} = 
N
$$

For non multiple of $N$ integers $k$, the numerator is clearly zero, thus we have

$$
\frac{1 - \exp(-2\pi i(k \text{ mod } 1))}{1 - \exp(-2\pi i(k \text{ mod } N)/N)} 
= \frac{\text{zero}}{\text{not zero}} 
= 0
$$


**(c)** 

A sine wave with wave-length $1/k$ is the imaginary componant of $\exp{2\pi ik_0x}$, a cos wave is the real componant. 

So the FFT of such a wave is

$$
\sum_{x=0}^{N-1}\exp(-2\pi i(k-k_0)x/N) = \frac{1-\exp(-2\pi i(k-k_0))}{1 - \exp(-2\pi i(k-k_0)/N)}
$$

TODO: plot here





