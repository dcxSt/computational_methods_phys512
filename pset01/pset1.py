import numpy as np

R12 = 1/12
R3 = 1/3

# Problem 1, testing
def ndiff4(f,x,d):
    return (- R12*f(x+2*d) + 2*R3*f(x+d) - 2*R3*f(x-d) + R12*f(x-2*d))/d

def ndiff2(f,x,d):
    return (f(x+d) - f(x-d))/(2*d)


# Problem 2
def ndiff(f, x, full=False):
    # Estimate optimal dx 
    dx = 6**(1/3) * 10**(-16/3) # Assumes f(x) is about 1
    # Derivative df/dx
    dfdx = (f(x+dx) - f(x-dx))/(2*dx)
    if full is True:
        # Estimate the error
        err = dx*dx/6
        return dfdx, dx, err
    return dfdx


# Problem 3, lakeshore diodes
def indicator(x1,x2):
    """Returns the indicator function Chi = 1 on [x1,x2)

    x1 : float
    x2 : float
    """
    def f(x):
        # Handle integers and floats
        if type(x) in (type(0.0),type(0)):
            if x>=x1 and x<x2: return 1.0
            else: return 0.0
        # Otherwise it's an ndarray
        out = np.zeros(x.shape)
        out[np.where(np.logical_and(x>=x1,x<x2))] = 1.0
        return out
    return f

def interp_linear(x,y):
    """Returns a linear interpolation function

    Demonstrates use of indicator functions. 

    Returns
    -------
    function
        Interpolation function that takes float, ints
        or ndarray and returns float or arrays.
    """
    assert x.shape == y.shape and len(x.shape) == 1
    def fun(xx):
        # Handle array and (float or int) input
        out = np.zeros(xx.shape) if type(xx)==np.ndarray else 0.0
        for x1,y1,x2,y2 in zip(x[:-1],y[:-1],x[1:],y[1:]):
            a = (y2-y1)/(x2-x1) # Slope between two points
            b = y1 - x1*a # Constant factor
            out += indicator(x1,x2)(xx) * (a*xx + b)
        return out
    return fun

            
def interp_cubic_spline(x,y):
    """Returns a function evaluates between x and y

    Parameters
    ----------
    x : ndarray
        The domain, x points
    y : ndarray
        The image of our function at the points x

    Returns
    -------
    function
        A cubic spline interpolation. 
    """

    # def cubic_spline()
    # 
    # return cubic_spline
    return 


# Interp 3d for a single point
def lakeshore(V, data):
    V_dat = data[:,0]
    T_dat = data[:,1]
    # TODO: this method is under construction
    print(V_dat)
    print(T_dat)
    

if __name__ == "__main__":
    print("INFO: testing ndiff")
    x = np.linspace(-50,10,5) # 10**(np.linspace(-5,1,7))
    dfdx, dx, err = ndiff(np.exp, x, True)
    print(f"INFO: x = {x}")
    print(f"INFO: d exp(x)/dx = {dfdx}")
    print(f"INFO: numerical - analytic = {np.exp(x)-dfdx}")
    print(f"INFO: estimated error = {err}")

    print("\nINFO: testing lakeshore")
    dat = np.loadtxt("lakeshore.txt")
    v = np.linspace(1,500,29)
    lakeshore(v, dat)





