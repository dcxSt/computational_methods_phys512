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
def interp3(x,y):
    """Returns a function that interpolates between x and y"""


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





