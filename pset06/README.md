
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

*Show that the Fourier transform of the window is* `[N/2,N/4,0,\cdots,0,N/4]`.

Let $w$ denote the window vector `0.5-0.5*np.cos(2*np.pi*np.arange(N)/N)`. Then it's FT is

$$
\begin{align*}
W \equiv Fw &= \sum_{x=0}^{N-1}\exp(-2\pi ix\xi/N) \left(\frac{1}{2} - \frac{1}{2}\cos(2\pi x/N)\right)\\
            &= \sum_{x=0}^{N-1} \exp(-2\pi ix\xi/N) \left(\frac{1}{2} - \frac{1}{4}\exp(-2\pi ix/N) - \frac{1}{4}\exp(2\pi ix/N) \right)\\
            &= \frac{1}{2}\sum_{x=0}^{N-1}\exp(-2\pi ix\xi/N) - \frac{1}{4}\sum_{x=0}^{N-1}\exp(-2\pi ix(\xi-1)/N) - \frac{1}{4}\sum_{x=0}^{N-1}\exp(-2\pi ix(\xi+1)/N)\\
            &= \delta(\xi)N/2 - \delta(\xi-1)N/4 - \delta(\xi+1\,\text{ mod }N)N/4
\end{align*}
$$

Therefore our array is `[N/2,-N/4,0,...,0,-N/4]`.

*Use this to show that you can get the windowed FT by appropriate combinations of each point in the unwindowed FT and it's immediate neighbors*

It follows from the convolution theorem tells us that the FT of pointwise multiplication of arrays $w$ and $x$ is the convolution of their fourier transforms.

$$
F^{-1}(W\ast X) = F^{-1}(W)\cdot F^{-1}(X) = w\cdot x\\
\Rightarrow F(w\cdot x) = W\ast X
$$

Since $W$ only has three terms in it, we can easily perform this convolution like so

```python
X=np.fft.fft(x)
windowed_fft_x = X/2-np.roll(X/4,1)-np.roll(X/4,-1)
```


## 5)

*Match Filter of LIGO data. We are going to find gravitational waves! Key will be getting LIGO data from github: [https://github.com/losc-tutorial/LOSC_Event_tutorial](https://github.com/losc-tutorial/LOSC_Event_tutorial).*

*While they include code to do much of this, please don't use it (although you may look at it for inspiration) and instead write your own. You can look at/use `simple_read_ligo.py` that I have posted for concise code to read the hdf5 files. Feel free to have your code loop over the events and print the answer to each part for that event. In order to make our life easy, in case we have to re-run your code, please also have a variable at the top of your code that sets the directory where you have unzipped the data. LIGO has two detectors (in Livingstone, Louisiana, and Hanford, Washington) and GW events need to be seen by both detectors to be considered rea. Note that my `read_template` functions returns the templates for both the plus and cross polarizations, but for simplicity you can just pick one of them to work with.*

**(a)**
*Come up with a noise model for the Livingston and Hanford detectors seperately. Describe in comments how you go about doing this. Please mention something about how you smooth the power spectrum and how you deal with lines (if at all). Please also explain how you window the data (you may wat to use a window that has an extended flat period near the center to avoid tapering the data/template where the signal is not small).*

We assume that the noise model is stationary

$$\langle f(x)f(x+\delta)\rangle = \langle g(\delta)\rangle$$

and then use the Wiener-Khinchin theorem, which says that the noise's power spectrum is the FT of $f$'s correlation function.

$$F\{N\} = |F\{S\}|^2$$

Therefore, the FT of the strain (abs squared) is our noise model. Let's implement:

We skip the `read_file` and `read_template` code that we have been give. First we import the windows that we need and establish a few conventional shortcuts.

```python
import numpy as np
from numpy.fft import rfft
import os
from os.path import join as pathjoin
import matplotlib.pyplot as plt
import h5py
import json
import scipy
from scipy.signal.windows import nuttall,hann,tukey,cosine,bartlett,blackman

# Custom window
flat=np.ones # we use this as moving-average kernel
root="./LOSC_Event_tutorial/"  # ligo data root directory (relative path)
```

Lets load and pack up the events metadata into a handy dictionary to use later

```python
# Load event metadata
events=json.load(open(pathjoin(root,'BBH_events_v3.json')))
# Fill dictionaries with all we need
hanford={}    # hanford detector data
livingston={} # livingston detector data
for e in events:
    # Hanford detector params
    strain,dt,utc = read_file(pathjoin(root,events[e]['fn_H1']))
    hanford[e]={"strain":strain,"dt":dt,"utc":utc}
    # Livingston detector params
    strain,dt,utc = read_file(pathjoin(root,events[e]['fn_L1']))
    livingston[e]={"strain":strain,"dt":dt,"utc":utc}
```

This function computes the PSD for us

```python
def get_psd(arr,window=None):
    """Get the psd of an array, optionally takes window func"""
    if window is not None:
        w_arr=window(len(arr)) # make window array
        arr_ft=rfft(arr*w_arr)
    else:
        arr_ft=rfft(arr)
    psd=np.abs(arr_ft)**2
    return psd
```

Now we compute and smooth the PSD for each event. We don't explicitly deal with the spikes, but instead, are mindful not to set the smoothing width to large so that the spikes aren't spread out too much. 

```python
window=hann # Select a windowing function to prevent leaking
width_smooth=20 # The width of the smoothing kernel
ker,ker_name=hann(width_smooth),"Hann" # smoothing kernel is window(width)
smooth = lambda x:np.convolve(x,ker,'same') # smoothing funciton

for e in events:
    # Load Hanford data
    strain,dt,utc = read_file(pathjoin(root,events[e]['fn_H1']))
    psd_h=get_psd(strain,window) # Hanford PSD
    # Load Livingston data
    strain,dt,utc = read_file(pathjoin(root,events[e]['fn_L1']))
    psd_l=get_psd(strain,window) # Livingston PSD
    # Plot them and compare (ommited)
```


![p5_smoothed_psd_GW150914](https://user-images.githubusercontent.com/21654151/201243063-148ea270-86ec-43e4-b803-28578379aa3d.png)
![p5_smoothed_psd_GW151226](https://user-images.githubusercontent.com/21654151/201243064-d7ca5087-35d4-4fe4-93ba-d46cbc65e570.png)
![p5_smoothed_psd_GW170104](https://user-images.githubusercontent.com/21654151/201243067-f9702923-3875-47a4-a034-91efe8d7aa9e.png)
![p5_smoothed_psd_LVT151012](https://user-images.githubusercontent.com/21654151/201243068-e0a42cf1-fbdf-4d68-a656-b68c1a25e68e.png)


**(b)**
*Use that noise model to search the four sets of events using a matched filter. The mapping between data and templates can be founmd in the file `BBH_events_v3.json`, included in the zipfile.*

**(c)**
*Estimate a noise for each event and from the output of the matched filter, give a signal-to-noise ratio for each event, both from the individual detectors, and from the combined Livingston + Hanford events.* 

**(d)**
*Compare the signal-to-noise you get from the scatter int he matched filter to the analytic signal-to-noise you expect from your noise model. How close are they? If they disagree, can you explain why?*














