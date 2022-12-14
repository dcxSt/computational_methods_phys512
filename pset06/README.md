
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
Code for this question is in jupyter notebook `p5.ipynb`

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
# Load event metadata
events=json.load(open(pathjoin(root,'BBH_events_v3.json')))

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

The matched filter works by cross-correlating a signal template with data. If our template is $A$ and $m$ is the (only) parameter of our model (template), and we have a good noise model $N$, then the chi-squared value of our parameter is

$$\chi^2=(d-Am)^TN^{-1}(d-Am)$$

We can exploit the stationarity of the noise to simplify this expression and solve for $m$

$$m(\tau) = (A(t-\tau)^TN^{-1}d)/(A(t-\tau)^TN^{-1}A(t-\tau))$$

We can ditch the denominator and simplify (which is constant, since $N$ is stationary) to obtain a relative most-likely amplitude

$$f \equiv (N^{-1}A(t-\tau))^Td \Rightarrow \tilde f = \tilde A^\ast \tilde N^{-1} \tilde d$$

These operations can be computed programatically like so

```python
# Load templates into variables
tp,tx = read_template(pathjoin(root,events[e]['fn_template']))
tp_win = tp*window(len(tp)) # Window template, we only use tp, not tx
tp_psd = np.abs(rfft(tp_win)**2) # Get PSD of template

# Noise models
ninv_h = 1/smooth(psd_h) # hanford N inv
ninv_l = 1/smooth(psd_l) # Livingston N inv

# Whiten the strains and template
sftwhite_h = sqrt(ninv_h)*rfft(strain_h) # whitend hanford strain FT
sftwhite_l = sqrt(ninv_l)*rfft(strain_l) # whitend livingston strain FT
tpftwhite_h = sqrt(ninv_h)*rfft(tp*window(len(tp)))# template whitned with Hanford noise model
tpftwhite_l = sqrt(ninv_l)*rfft(tp*window(len(tp)))# template whitned with Livingston noise model

# Matched filters
mh = irfft(np.conj(tpftwhite_h) * sftwhite_h) # Hanford matched filter
ml = irfft(np.conj(tpftwhite_l) * sftwhite_l) # Livingston matched filter
```

NB: `strain_h` is windowed, and the `tukey` window is used in below plots. 


![p5_matched_filter_GW150914](https://user-images.githubusercontent.com/21654151/201450016-ce54c581-7a3f-470a-badc-4b4b094d44ef.png)
![p5_matched_filter_GW151226](https://user-images.githubusercontent.com/21654151/201450018-1778adfb-7da7-4509-a438-7816ec7473c5.png)
![p5_matched_filter_GW170104](https://user-images.githubusercontent.com/21654151/201450021-378e1df3-c121-4e5a-a493-399903dae6f5.png)
![p5_matched_filter_LVT151012](https://user-images.githubusercontent.com/21654151/201450024-9aa5ffba-1424-4ef6-9957-9b1c93726bdd.png)




**(c)**
*Estimate a noise for each event and from the output of the matched filter, give a signal-to-noise ratio for each event, both from the individual detectors, and from the combined Livingston + Hanford events.* 

The variance is

$$\langle\delta_m\delta_m^T\rangle = (A^TN^{-1}A)^{-1}$$

In our case, $m$ is a scalar and so is $\delta_m$. We can use the fact that the noise is stationary so fourier transforms will cast $N$ into a diagonal form

$$\text{Var}(m) = ((FA)^\dagger(FNF^\dagger)^{-1}(FA))^{-1} = (FA/\sqrt{1/PSD})^\dagger (FA/\sqrt{PSD}))^{-1}$$

The rightmost innequality is obtained with the Wiener Kinchen theorem.

The signal to noise ratio is $P_{s}/P_{n}$, the power of the signal divided by the power of the noise, (we use the square root of that). Our signal is a constant, which we estimate by taking the max of the match filter. Our noise is a random variable, the power is the expected value of $N^2$. 

We can find an analytic expression for what we expect the noise to be based only on our templates and noise model

$$(A^TN^{-1}A)^{-1/2}$$


```python
### (c)
# Estimate noise analytically
AtNinvA_h = 2*abs(tpftwhite_h@tpftwhite_h) # factor of 2 from rfft
AtNinvA_l = 2*abs(tpftwhite_l@tpftwhite_l) # factor of 2 from rfft

# Estimate signal to noise ratio
snr_anal_h = max(abs(mh))*sqrt(AtNinvA_h) # signal to noise analytic hanford
snr_anal_l = max(abs(ml))*sqrt(AtNinvA_l) # " Livingston
snr_esti_h = max(abs(mh))/np.std(mh[:35000]) # std is sq power since mean is zero
snr_esti_l = max(abs(ml))/np.std(ml[:35000]) # std is sq power since mean is zero

# Display SNR estimates
print(f"{e}")
print(f"SNR analitic estimate hanford/livingston={snr_anal_h:.2f}/{snr_anal_l:.2f}")
print(f"SNR measured estimate hanford/livingston={snr_esti_h:.2f}/{snr_esti_l:.2f}")
```

**(d)**
*Compare the signal-to-noise you get from the scatter in the matched filter to the analytic signal-to-noise you expect from your noise model. How close are they? If they disagree, can you explain why?*

The output is dubious for the analytically computed one
```
GW150914
SNR analitic estimate hanford/livingston=134.32/48.99
SNR measured estimate hanford/livingston=17.39/12.79
LVT151012
SNR analitic estimate hanford/livingston=39.70/34.89
SNR measured estimate hanford/livingston=6.45/5.11
GW151226
SNR analitic estimate hanford/livingston=9.54/7.31
SNR measured estimate hanford/livingston=9.94/6.95
GW170104
SNR analitic estimate hanford/livingston=47.50/73.65
SNR measured estimate hanford/livingston=8.00/9.47
```


The analytic estimate drastically understimates the noise (but not too drastically, it's not more than one order of magnitude). This is probably because we smoothed those spikes in our noise model, so the analytical formula is underestimating the noise. 

Perhaps we should put more effort into a noise model. To deal with noise spikes, we might use a peak detection algorithm, then fit a skinny gaussian to each peak, and then add those to a smoothened signal without the spikes. 


**(e)**
*From the template and noise model, find the frequency from each event where half the weight comes from above that frequency and half below.*

To callibrate data points with frequency, we define a frequency array

```python
sr = 1/dt # sample rate
nyquist = sr/2
freq = np.linspace(0,nyquist,len(strain)//2+1) # //2+1 to account for rfft
```

Then we take the absoulte rFFT of the whitened template and find the 0.5 crossing point.

```python
# The normalized cumulative frequency weight
tpwhite_cum = np.cumsum(abs(tpftwhite))/sum(abs(tpftwhite))
# The 0.5 crossing
freqhalf = freqs[np.argwhere(tpwhite_cum>0.5).min()]
```

With one exception, these frequencies agree to about one sig fig between the Hanford and Livingston detectors. 


![p5_cumulativeGW150914](https://user-images.githubusercontent.com/21654151/201718830-087d9889-7d3d-478a-879f-fcb90233739e.png)
![p5_cumulativeGW151226](https://user-images.githubusercontent.com/21654151/201718831-f469127f-f5c6-4961-8096-696d25b943ed.png)
![p5_cumulativeGW170104](https://user-images.githubusercontent.com/21654151/201718833-b97d40d6-f2a4-43de-861c-d60ab3d55f41.png)
![p5_cumulativeLVT151012](https://user-images.githubusercontent.com/21654151/201718834-56c956ab-ae8b-4a1f-8c64-96cafc59b8b2.png)


**(f)**
*How well can you localize the time of arrival (the Horizontal shoft of your matched filter). The positions of gravitational wave events are infered by comaring their arrival times at different detectors. What is the typical positional uncertainty you might expect given that the detectors are a few thousand km apart?*



Graviational travel at the speed of light $c\approx 3\cdot 10^8m/s$. Lets say that the detectors are 3000 km appart. It takes light $10^{-2}$ seconds, or 10 ms to travel that distance. If the egent happens on the plane, which is the perpendicular bisector to the line that joins the detectors, the signal will arrive at the same time to each detector. With this image, we can deduce that the angle of arival is determined by this equation

$$T = L\cos\theta / c \approx 10ms \cos\theta \Rightarrow \theta = \arccos\frac{T}{10ms}$$

Above, $\theta$ is the angle that the line comming from the source makes with the line that joins Livingston and Hanford. If our error in $T$ is $\Delta T$, we can approximate the error in $\theta$ by taylor expanding

$$\cos(\theta\pm\Delta\theta) \approx \cos\theta \mp\Delta\theta\sin\theta \approx (T\pm\Delta T)/10ms$$

Canceling $\cos\theta=T/10ms$ we get

$$\Delta \theta \approx \Delta T / (10ms \cdot \sin\theta)$$

When $\theta\approx 0$, we need a second order taylor expansion, and we get

$$\Delta \theta \approx \begin{cases}\frac{\Delta T}{10ms\cdot \sin\theta},\qquad\text{when $\sin\theta$ is not zero}\\ \sqrt{\frac{2\Delta T}{10ms}},\qquad\text{when $\theta\approx 0$}\end{cases}$$

Roughly speaking, the positional uncertainty will be about $\Delta \theta\approx \Delta T/10ms$. We estimate our $\Delta T$ by the half-width of the region for which the signal to noise ratio is within one of the peak. Typically these are about $\Delta T=0.5ms$, so $\Delta \theta\approx 0.05$ radians. (However, it is much worse in LVT151012 where $\Delta T$ is larger and $\theta\approx\pi/2$)


![arrival_times_GW150914](https://user-images.githubusercontent.com/21654151/201743040-7cb948d9-3626-48f6-9f6f-54fa6b355487.png)
![arrival_times_GW151226](https://user-images.githubusercontent.com/21654151/201743045-65fba4b0-f72e-4ac6-ac4c-67a8e7e534a4.png)
![arrival_times_GW170104](https://user-images.githubusercontent.com/21654151/201743047-b11e8656-115c-4acd-8128-60bf14d4e8eb.png)
![arrival_times_LVT151012](https://user-images.githubusercontent.com/21654151/201743049-9d391aa4-f55d-4ab4-8909-d3ca9a03dcfa.png)

