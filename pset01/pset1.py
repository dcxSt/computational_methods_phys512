import numpy as np

R12 = 1/12
R3 = 1/3

# Problem 1, testing
def ndiff4(f,x,d):
    return (- R12*f(x+2*d) + 2*R3*f(x+d) - 2*R3*f(x-d) + R12*f(x-2*d))/d

def ndiff2(f,x,d):
    return (f(x+d) - f(x-d))/(2*d)

# Problem 2
def ndiff(f,x,full=False):
    # Estimate optimal dx 
    dx = 6**(1/3) * 10**(-16/3) # Assumes f(x) is about 1
    # Derivative df/dx
    dfdx = (f(x+dx) - f(x-dx))/(2*dx)
    if full is True:
        # Estimate the error
        err = dx*dx/6
        return dfdx, dx, err
    return dfdx

# Problem 4








