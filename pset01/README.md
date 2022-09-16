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



## Problem 4, Interpolation
Take `cos(x)` between `-pi` and `pi`. Compare the accuracy of polynomial, cubic spline, and rational function interpolation given some modest number of points, but for fairness each method should use the same points. Now try using a Lorentzian `1/(1+x*x)` between `-1` and `1`. 



*What should the error be for the Lorentzian from the rational function fit? Does what you got agree with the expectations when the order is higher (say `n=4, m=5`)? What happens if you switch from `np.linalg.inv` to `np.linalg.pinv` (which tries to deal with singular matrices)? Can you understand what has happend by looking at `p` and `q`? As a hint, think about why we had to fix the constant term in the denominator, and how that might generalize.* 

The error for the Lorentzian from the rational function fit should be pretty damn close to the roundoff error. This is because a low order rational fit will find the exact (up to roundoff) parameters of the Lorentzian. 

However when the order is higher, the matrix that we invert to solve for those values will be degenerate, because the coefficients are over-specified. Zero eigenvalues leads to infinite eigenvalues in the inverse.

Switching to `np.linalg.pinv` solves the problem as this routine is designed to set those eigenvalues that blow up, to zero. 

Can you understand what has happend by looking at `p` and `q`? Yes. The coefficients of `p` and `q` are over specified. That is to say, if we factor out `1/(x**2+1)` from `q`, then the remaining coefficients of `q` can be anything so long as they cancel out with the coefficients of `p` on top. Since `x**2+1` is order two, this means we have `min(n,m-2)` degrees of freedom, i.e. `min(n,m-2)` zero eigenvalues. 
















