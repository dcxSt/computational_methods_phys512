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
### BEGIN: Tried & failed to write my own cubic spline 3
# This function works
def get_indicator(x1,x2):
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

# This function works
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
            out += get_indicator(x1,x2)(xx) * (a*xx + b)
        return out
    return fun

# This function doesn't work!
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
    spline_params=[] # list of functions on segments
    indicators=[] # list of indicator functions 
    yip=(y[1]-y[0])/(x[1]-x[0]) # First derivative at x[i]
    yipp=0.0 # Second derivative of y at x[i]
    for x1,y1,x2,y2 in zip(x[:-1],y[:-1],x[1:],y[1:]):
        dx=x2-x1 # centor x a x1
        rdx=1/dx
        d=y1 # constant term
        c=yip # First derivative
        b=0.5*yipp # Second derivative
        a=rdx**3*y2-rdx*b-rdx**2*c-rdx**3*d
        # def spline_segment(x):
        #     xs=x-x1 # shift x
        #     return a*xs**3+b*xs**2+c*xs+d
        # Compute derivatives for the next round
        yip=3*a*dx**2+2*b*dx+c
        yipp=6*a*dx+2*b
        # Add spline params to list
        spline_params.append((a,b,c,d,x1,rdx))
        indicators.append(get_indicator(x1,x2))

    # Construct the Cubic spline
    def cubic_spline(xx):
        out=np.zeros(xx.shape) if type(xx)==np.array else 0.0       
        # set ypp and y initial conditions
        for indicator,(a,b,c,d,x1,rdx) in zip(indicators,spline_params):
            xs = xx-x1 # Shift x by x1
            out += indicator(xx)*a*xs**3+b*xs**2+c*xs+d
            # out += spline(xx)*indicator(xx)
        return out
    
    return cubic_spline


# Interp 3d for a single point
def lakeshore(V, data):
    V_dat = data[:,0]
    T_dat = data[:,1]
    # TODO: this method is under construction
    print(V_dat)
    print(T_dat)
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("DEBUG: testing ndiff")
    x = np.linspace(-50,10,5) # 10**(np.linspace(-5,1,7))
    dfdx, dx, err = ndiff(np.exp, x, True)
    print(f"DEBUG: x = {x}")
    print(f"DEBUG: d exp(x)/dx = {dfdx}")
    print(f"DEBUG: numerical - analytic = {np.exp(x)-dfdx}")
    print(f"DEBUG: estimated error = {err}")

    print("\nDEBUG: testing lakeshore")
    dat = np.loadtxt("lakeshore.txt")
    v = np.linspace(1,500,29)
    lakeshore(v, dat)

    print("\nDEBUG: testing cubic spline")
    fun=np.cos
    x=np.linspace(-2*np.pi,2*np.pi,6)
    y=fun(x)
    # Get the cubic spline
    spline=interp_cubic_spline(x,y)
    xfine=np.linspace(-2*np.pi,2*np.pi,1000)
    yspline=spline(xfine)
    print(f"DEBUG: {yspline}")
    ytrue=fun(xfine)
    plt.figure()
    plt.plot(x,y,"o", label="samples interpolated")
    plt.plot(xfine,ytrue,label="true")
    plt.plot(xfine,yspline,label="spline")
    plt.legend()
    plt.show(block=True)








