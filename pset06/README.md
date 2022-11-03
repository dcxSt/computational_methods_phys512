
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

![p1_gaussian](https://user-images.githubusercontent.com/21654151/199855804-3d7692ec-3de8-455c-aeb4-a057f72eb3d3.png)


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

![p1_correlations_of_gaussians](https://user-images.githubusercontent.com/21654151/199855822-f2230cbe-e1ed-4d13-aa59-eaaa5b7ec0ee.png)


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
\lim_{k\to 0}\frac{2\pi i \exp(-2\pi ik)}{2\pi i/N \exp(-2\pi i k/N)} = 
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

![freqency_leaking](https://user-images.githubusercontent.com/21654151/199855882-b2a6376c-4296-422e-bb08-a8da58cde4a1.png)


**(d)**

*A common tool to get around leakage is the use of window functions. The leakage essentially comes from the fact that we have a sharp jump at the edge of the inveral. If we multiply our input data by a function that goes to zero at the edges, this cancels out the jump, and so prevents the leakage from the jumpas at the edges. Of course, since we have multiplied by the window in real space, we have convolved by it in Fourier space. Once simple window we could use is $0.5-0.5\cos(2\pi x/N)$ (there are many, many choices). Show that when we multiply by this window, the spectral leakage for a non-integer period sine wave drops dramatically.*

![freqency_leaking_windowed](https://user-images.githubusercontent.com/21654151/199855917-da37f26d-10b6-4f14-8b38-6a3f8c08d7b0.png)


**(e)**

*Show that the Fourier transform of the window is* $[N/2,N/4,0,\cdots,0,N/4]$

Let $w$ denote the window vector `0.5-0.5*np.cos(2*np.pi*np.arange(N)/N)`. Then it's FT is

$$
\begin{align*}
W \equiv Fw &= \sum_{x=0}^{N-1}\exp(-2\pi ix\xi/N) \left(\frac{1}{2} - \frac{1}{2}\cos(2\pi x/N)\right)\\
            &= \sum_{x=0}^{N-1} \exp(-2\pi ix\xi/N) \left(\frac{1}{2} - \frac{1}{4}\exp(-2\pi ix/N) - \frac{1}{4}\exp(2\pi ix/N) \right)\\
            &= \frac{1}{2}\sum_{x=0}^{N-1}\exp(-2\pi ix\xi/N) - \frac{1}{4}\sum_{x=0}^{N-1}\exp(-2\pi ix(\xi-1)/N) - \frac{1}{4}\sum_{x=0}^{N-1}\exp(-2\pi ix(\xi+1)/N)\\
            &= \delta(\xi)N/2 - \delta(\xi-1)N/4 - \delta(\xi+1\,\text{ mod }N)N/4
\end{align*}
$$

It follows from the convolution theorem tells us that the FT of pointwise multiplication of arrays $w$ and $x$ is the convolution of their fourier transforms.

$$
F(w\cdot x) = W\ast X
$$

Since $W$ only has three terms in it, we can easily perform this convolution like so

```python
X=np.fft.fft(x)
windowed_fft_x = X/2-np.roll(X/4,1)-np.roll(X/4,-1)
```












