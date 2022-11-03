import numpy as np
from numpy.fft import fft

PI=np.pi

def analytic_dft_sinewave_k(k0:float,N:int):
    """Takes the DFT of a sinewave of wavelength k0, arrlength n"""
    k=np.arange(N)
    fftsine=1-np.exp(-2.0j*PI*(k-k0))
    fftsine/=1-np.exp(-2.0j*PI*(k-k0)/N)
    return fftsine

def numpy_dft_sinewave_k(k0:float,N:int,window:np.ndarray=None):
    """Takes the DFT of a sinewave of wavelength k0, arrlength n"""
    k=np.arange(N)
    sine=np.exp(2.0j*PI*k0*np.arange(N)/N)
    if isinstance(window,type(None)):
        return fft(sine) # If no window, return fft
    return fft(sine*window)



if __name__=="__main__":
    import matplotlib.pyplot as plt

    # Question 4, DFTs
    k0s=[16.1,16.5,32.3,32.9,128.25,128.3]
    colors=["g","r","b","k","yellow","orange"]#["g","r","b"]
    N=256
    # Leakage without windows
    plt.figure(figsize=(10,5))
    for k0,c in zip(k0s,colors):
        analdft=analytic_dft_sinewave_k(k0,N)
        plt.semilogy(np.abs(analdft),"--",color=c,alpha=0.5,label=f"anal k0={k0}")
        npdft  =numpy_dft_sinewave_k(k0,N) # Numpy dft
        plt.semilogy(np.abs(npdft),".",color=c,label=f"numpy k0={k0}")
        #window =np.cos(np.linspace(-PI/2,PI/2,N))
        #windft =numpy_dft_sinewave_k(k0,N,window) # windowed dft
        #plt.semilogy(np.abs(windft),"-.",color=c,label=f"windowed k0={k0}")
        plt.title("Frequency Leaking",fontsize=20)
    plt.grid(which="both")
    plt.xlabel("Fourier mode",fontsize=16)
    plt.ylabel("Amplitude",fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./img/freqency_leaking.png",dpi=450)
    plt.show(block=True)

    # Leakage with windows
    plt.figure(figsize=(10,5))
    for k0,c in zip(k0s,colors):
        analdft=analytic_dft_sinewave_k(k0,N)
        plt.semilogy(np.abs(analdft),"--",color=c,alpha=0.5,label=f"anal k0={k0}")
        npdft  =numpy_dft_sinewave_k(k0,N) # Numpy dft
        plt.semilogy(np.abs(npdft),".",color=c,label=f"numpy k0={k0}")
        #window =np.cos(np.linspace(-PI/2,PI/2,N)) # half-cos window
        window =0.5-0.5*np.cos(2*PI*np.arange(N)/N)
        windft =numpy_dft_sinewave_k(k0,N,window) # windowed dft
        plt.semilogy(np.abs(windft),"x",color=c,label=f"windowed k0={k0}")
        plt.title("Frequency Leaking",fontsize=20)
    plt.grid(which="both")
    plt.xlabel("Fourier mode",fontsize=16)
    plt.ylabel("Amplitude",fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./img/freqency_leaking_windowed.png",dpi=450)
    plt.show(block=True)






