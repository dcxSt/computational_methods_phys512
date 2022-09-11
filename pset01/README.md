Due Friday Sep 16 at 11:59PM

## Problem 1, Taking a derivative



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




## Problem 4, Interpolation




