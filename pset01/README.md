Due Friday Sep 16 at 11:59PM

## Problem 1, Taking a derivative

![solution1_stephenfay](https://user-images.githubusercontent.com/21654151/189554653-773efb91-20eb-4758-9d39-75507014af1c.png)

## Problem 2, Write a numerical differentiator prototype
```python
def ndiff(f, x, full=False):
  # Estimate optimal dx
  dx = 6*(1/3) * 10**(-16/3) # Assume f(x) is about 1
  # Derivative df/dx
  dfdx = (f(x+dx) - f(x-dx)) / (2*dx)
  if full is True:
    # Estimate the error, and return f', dx, error
    err = dx*dx/6
    return dfdx, dx, err
  return dfdx
```

## Problem 3, Lakeshore 670 diodes

We interpolate with a cubic spline, only using the nearest two data points, and taking advantage of the fact we are given the derivatives.

```python
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
        tp2=1/volt[idx2]
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
        tfit,logerr = interp(val,volt,temp,dvdt)
        tfit_arr.append(tfit)
        logerr_arr.append(logerr)
    print("DEBUG: returning")
    return tfit_arr,logerr_arr
```




## Problem 4, Interpolation
Take `cos(x)` between `-pi` and `pi`. Compare the accuracy of polynomial, cubic spline, and rational function interpolation given some modest number of points, but for fairness each method should use the same points. Now try using a Lorentzian `1/(1+x*x)` between `-1` and `1`. 



*What should the error be for the Lorentzian from the rational function fit? Does what you got agree with the expectations when the order is higher (say `n=4, m=5`)? What happens if you switch from `np.linalg.inv` to `np.linalg.pinv` (which tries to deal with singular matrices)? Can you understand what has happend by looking at `p` and `q`? As a hint, think about why we had to fix the constant term in the denominator, and how that might generalize.* 

The error for the Lorentzian from the rational function fit should be pretty damn close to the roundoff error. This is because a low order rational fit will find the exact (up to roundoff) parameters of the Lorentzian. 

However when the order is higher, the matrix that we invert to solve for those values will be degenerate, because the coefficients are over-specified. Zero eigenvalues leads to infinite eigenvalues in the inverse.

Switching to `np.linalg.pinv` solves the problem as this routine is designed to set those eigenvalues that blow up, to zero. 

Can you understand what has happend by looking at `p` and `q`? Yes. The coefficients of `p` and `q` are over specified. That is to say, if we factor out `1/(x**2+1)` from `q`, then the remaining coefficients of `q` can be anything so long as they cancel out with the coefficients of `p` on top. Since `x**2+1` is order two, this means we have `min(n,m-2)` degrees of freedom, i.e. `min(n,m-2)` zero eigenvalues. 
















