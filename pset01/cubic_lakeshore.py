import numpy as np

def interp_cubic(v,volt,temp,dvdt):
    """Interpolate the temp as function of volt, evaluate on v

    Assumes volt is strictly increasing.

    v : ndarray or float or int
        evaluate T at these(this) value
    volt : ndarray
        The voltage measurments we interp between. Assume this array is 
        decreasing, as this is the data we are given. 
    temp : ndarray
        The temperature measurements to interp.
    dvdt : ndarray
        The dvdt        
    """
    if type(v)!=np.ndarray:
        v=float(v)
        idx1=max(np.where(volt<=v)[0])
        idx2=idx1+1
        v1=volt[idx1]
        v2=volt[idx2]
        t1=temp[idx1]
        t2=temp[idx2]
        tp1=1/dvdt[idx1]
        tp2=1/dvdt[idx2]
        dv=v2-v1
        # These numbers come from the pretty derivation, see paper
        mat=np.array([[dv**2/2 , dv**3/6],
                      [dv      , dv**2/2]])
        minv=np.linalg.inv(mat)
        tpp1,tppp1 = minv@np.array([t2-t1-tp1*dv, tp2-tp1])
        # Evaluate t(v)
        tfit=t1+tp1*(v-v1)+tpp1*(v-v1)**2/2+tppp1*(v-v1)**3/6
        # Estimate the error, if one point dominates, then the 
        # L2 norm converges to L-infinity norm, so we take max
        # instead of bothering ourselves with square roots
        logepsilon=-16
        logerr=max(4*np.log10(dv/2)-np.log10(4*3*2),logepsilon)
        return tfit,logerr
    # Otherwise, loop through, recursive call above
    tfit_arr,logerr_arr=[],[]
    for i,val in enumerate(v):
        print(f"DEBUG: i={i}, v={v}")
        tfit,logerr = interp_cubic(val,volt,temp,dvdt)
        tfit_arr.append(tfit)
        logerr_arr.append(logerr)
    print("DEBUG: returning")
    return tfit_arr,logerr_arr

       


if __name__=="__main__":
    dat=np.loadtxt("lakeshore.txt")
    dat=dat[dat[:, 1].argsort()] # Sort ascending volt
    print(f"DEBUG: dat.shape={dat.shape}")
    temp=dat[:,0]
    volt=dat[:,1]
    dvdt=dat[:,2]*0.001 # Normalize units

    print("DEBUG: check1")
    voltfine=np.linspace(volt[1],volt[-2],1001)
    tempinterp,logerr=interp_cubic(voltfine,volt,temp,dvdt)
    
    print("DEBUG: check2")
    tempinterp=np.array(tempinterp)
    logerr=np.max(np.array(logerr))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.title("Cubic interpolation using two points\nand the derivatives at those points")
    plt.plot(volt,temp,"o",label="values given")
    plt.plot(voltfine,tempinterp,label=f"interpolation logerr estimate: {logerr}")
    plt.legend()
    plt.xlabel("V")
    plt.ylabel("T")
    plt.savefig("plots/cubic_lakeshore.png")
    plt.show(block=True)









