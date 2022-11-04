import numpy as np
from numpy.fft import rfft,irfft


def dftshift(arr:np.ndarray, x0:int):
    """shifts arr by x0, f(x)-->f(x-x0)
    Only works for even lengthed arrays, to make use of irfft.
    Does the same thing as np.roll(arr,x0)"""
    assert len(arr)%2==0, f"array length {arr.shape} must be even 1-d"
    N=len(arr)
    arrft=rfft(arr)
    twidle=np.exp(-2.0j*np.pi*x0/N*np.arange(0,N//2+1))
    shifted=irfft(arrft*twidle)
    return shifted

def correlation(u,v):
    """Returns correlation of u and v"""
    return irfft(rfft(u)*np.conj(rfft(v)))

def correlation_pad(u,v):
    """Returns correlation of u and v without any wrap-around"""
    zeros=np.zeros(len(u)) # assume len(u)==len(v)
    upad,vpad=np.hstack((u,zeros)),np.hstack((v,zeros))
    return correlation(upad,vpad)

def autocorr_shifted(u, shift):
    """correaltion of a function with a shifted version of it's self"""
    u_shifted=np.roll(u,shift)
    return correlation(u,u_shifted)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x=np.linspace(-5,5,256)
    arr=np.exp(-x**2) # a gaussian function

    # TEST dftshift
    shift=len(arr)//4#len(arr)//2
    shifted_arr=dftshift(arr,shift)
    plt.figure(figsize=(7,4))
    plt.plot(x,arr,label="arr")
    plt.plot(x,shifted_arr,label="shifted")
    plt.plot(x,np.roll(arr,shift),"--",label="shifted np.roll")
    plt.legend()
    plt.title("Shifted gaussian")
    plt.savefig("./img/p1_gaussian.png",dpi=450)
    plt.show(block=False)
    plt.pause(0.2)

    # Correlation of gaussian with it's self
    plt.figure(figsize=(7,4))
    plt.plot(correlation(arr,arr),label="autocorr gaussian")
    shifts=(16,32,64,128)
    for shift in shifts:
        plt.plot(autocorr_shifted(arr,shift),label=f"shifted by {shift}")
    plt.title("Correlations of gaussians with themselves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./img/p1_correlations_of_gaussians.png")
    plt.show(block=True)
    plt.pause(0.2)

    # Test correlation pad
    # TODO
 

   

