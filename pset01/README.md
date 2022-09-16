Due Friday Sep 16 at 11:59PM
Made an effort to make things clean and tidy! Look at those comments! Please give all the bonus points!

## Problem 1, Taking a derivative

![solution1_stephenfay](https://user-images.githubusercontent.com/21654151/189554653-773efb91-20eb-4758-9d39-75507014af1c.png)

## Problem 2, Write a numerical differentiator prototype
NB: this works with `x` as an array so long as `f` takes array arguments. 
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

![lakeshore_spline_with_derivative](https://user-images.githubusercontent.com/21654151/190710120-34775a0f-dcca-4c26-aceb-3aa7c1468f20.png)


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

As indicated in the comments in the code above, we estimate the ballpark error by taking the max of the trunkation error and the system error. 

![cubic_lakeshore](https://user-images.githubusercontent.com/21654151/190709724-b7ba350b-9420-4405-8dd3-cc6ff91d1149.png)

We have to be careful to normalize before calling the interpolation function. If testing code above, see `cubic_lakeshore.py` in this directory. 

```python
dat=np.loadtxt("lakeshore.txt")
dat=dat[dat[:,1].argsort()] # Sort ascending volt
temp=dat[:,0]
volt=dat[:,1]
dvdt=dat[:,2]*0.001 # Normalize to correct units
```

## Problem 4, Interpolation
*Take `cos(x)` between `-pi` and `pi`. Compare the accuracy of polynomial, cubic spline, and rational function interpolation given some modest number of points, but for fairness each method should use the same points. Now try using a Lorentzian `1/(1+x*x)` between `-1` and `1`.* 

First, here is the code for the rational and poly fits. And we steal a better cubic interpolation from scipy (that knows how to deal with edge-points) `scipy.interpolate.interp1d`. 

```python
import numpy as np

def rational(coeffs,x,n:int,m:int):
    """Evaluates the rational function p(x)/(1+qq(x))

    coeffs : ndarray
        The coefficients of p(x) and qq(x)  

    x : ndarray or float/int
        Data to evaluate.
    n : int
        Order of numerator polynomial.
    m : int
        Order of denomenator polynomial.
    """
    assert coeffs.shape==(n+m-1,) # Sanity check
    coeffs_numer = coeffs[:n]
    coeffs_denom = np.hstack([[1],coeffs[n:]])
    numer = np.vstack([x**i for i in range(n)]).T@coeffs_numer
    denom = np.vstack([x**i for i in range(m)]).T@coeffs_denom
    return numer/denom

def polynomial(coeffs,x):
    """Evaluates the polynomial function w/ coefficients `coeffs` at x
    
    coeffs : ndarray
        The ordered coefficients of the polynomial, from x^0 up.
    x : ndarray or float/int
        Data to evaluate

    Returns
    -------
    ndarray or float/int
        p(x) = dot(coeffs,x)
    """
    ord=coeffs.shape[0]-1 # -1 ugly but necessary for consistancy
    y=np.vstack([x**k for k in range(ord+1)]).T@coeffs
    return y


def rational_fit(x:np.ndarray, y:np.ndarray, n:int, m:int):
    """Fit the data to a rational function using linear algebra

    The rational fit will have an order n polynomial on the numerator
    and an order m polynomial with constant term 1 on the denomenator.
    """
    # Sanity checks, easier debugging and *reading* (hint hint TA's)
    assert x.shape==y.shape, "You're up to no good my friend"
    assert x.shape==(n+m-1,), f"Too many sample points to fit, {x.shape} must match n+m-1={n+m-1}, try least squares fit instead."
    # Build the part of the matrix corresponding to p(x)
    mat_p=np.zeros((n+m-1,n))
    for k in range(n):
        mat_p[:,k] = x**k
    # Build the part of the matrix corresponding to qq(x)
    mat_qq=np.zeros((n+m-1,m-1))
    for k in range(1,m):
        mat_qq[:,k-1] = -y * (x**k)
    # Stack the rectangular 'parts' to get a square matrix
    mat=np.hstack((mat_p,mat_qq))
    # Invert the matrices, solve for the coefficients
    inv=np.linalg.pinv(mat)
    coeffs=inv@y
    return coeffs

def polynomial_fit(x:np.ndarray, y:np.ndarray, ord:int):
    """Fit the data to a polynomial using linear algebra

    Parameters
    ----------
    x : np.ndarray
        The x data, must be 1d array with shape=(ord+1,)
    y : np.ndarray
        The y data, must be 1d array with shape=(ord+1,)
    ord : int
        The order of the polynomial we are fitting exactly to the data

    Returns
    -------
    np.ndarray
        Coefficients. ndarray with shape=(ord+1,)
        i.e. P(x)=np.dot(coeffs,[x**k for k in range(ord+1)])
    """
    assert x.shape==y.shape
    assert x.shape==(ord+1,)
    mat=np.zeros((ord+1,ord+1))
    for k in range(ord+1): mat[:,k]=x**k
    inv=np.linalg.pinv(mat)
    coeffs=inv@y
    return coeffs
```

Here are plots of how each of these methods performs on the cosine function. 

![fits4_n=3_m=2](https://user-images.githubusercontent.com/21654151/190714060-62cbb28b-8669-45b4-b201-de5501a16435.png)
![fits4_n=3_m=3](https://user-images.githubusercontent.com/21654151/190714062-833b21da-7837-45a1-8b76-ea2ff08c3dd9.png)
![fits4_n=6_m=5](https://user-images.githubusercontent.com/21654151/190714063-07449d00-c6bd-4a82-84ba-f7388afccddf.png)
![fits4_n=7_m=8](https://user-images.githubusercontent.com/21654151/190714065-28d064b1-623d-4a2b-9c04-5fe1992efdd1.png)
![fits4_n=9_m=8](https://user-images.githubusercontent.com/21654151/190714066-cac80bb7-0812-4204-8f68-e61965e83ee6.png)
![fits4_n=12_m=11](https://user-images.githubusercontent.com/21654151/190714067-c2775059-a602-4bc9-8dce-73d7b78eff12.png)

Here are plots of how each of these methods performs on the laplacian, provided we use `np.linalg.pinv` rather than `np.linalg.inv` to invert our degenerate matrices. 

![fits4_laplacian_n=2_m=3](https://user-images.githubusercontent.com/21654151/190715851-aee684fa-9242-4abc-927c-44f7262b399f.png)
![fits4_laplacian_n=3_m=3](https://user-images.githubusercontent.com/21654151/190715852-52418823-a4f4-463d-b96f-79b2ec521c0d.png)
![fits4_laplacian_n=6_m=6](https://user-images.githubusercontent.com/21654151/190715854-ac0ff864-a788-4f4d-8a95-a3c10feb1d59.png)
![fits4_laplacian_n=7_m=8](https://user-images.githubusercontent.com/21654151/190715856-64e7890d-f5e5-443d-85e0-deeaa2c5dc83.png)


*What should the error be for the Lorentzian from the rational function fit? Does what you got agree with the expectations when the order is higher (say `n=4, m=5`)? What happens if you switch from `np.linalg.inv` to `np.linalg.pinv` (which tries to deal with singular matrices)? Can you understand what has happend by looking at `p` and `q`? As a hint, think about why we had to fix the constant term in the denominator, and how that might generalize.* 

The error for the Lorentzian from the rational function fit should be pretty damn close to the roundoff error. This is because a low order rational fit will find the exact (up to roundoff) parameters of the Lorentzian. 

However when the order is higher, the matrix that we invert to solve for those values will be degenerate, because the coefficients are over-specified. Zero eigenvalues leads to infinite eigenvalues in the inverse.

Switching to `np.linalg.pinv` solves the problem as this routine is designed to set those eigenvalues that blow up, to zero. 

Can you understand what has happend by looking at `p` and `q`? Yes. The coefficients of `p` and `q` are over specified. That is to say, if we factor out `1/(x**2+1)` from `q`, then the remaining coefficients of `q` can be anything so long as they cancel out with the coefficients of `p` on top. Since `x**2+1` is order two, this means we have `min(n,m-2)` degrees of freedom, i.e. `min(n,m-2)` zero eigenvalues. 
















