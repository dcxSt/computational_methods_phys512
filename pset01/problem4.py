import numpy as np
from scipy.interpolate import interp1d

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



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # The function we are fitting
    fun=lambda x:1/(1+x**2) # Laplacian # fun=np.cos
    funcname="laplacian" # This is for the plot
    
    # Rational function order of numer and denom polynomials
    n,m=6,6
    ord=n+m-2 # Order of polynomial fit
    # Number of degrees of freedom / number of coeffs to fit is ord+1
    
    # Set a modest number of points
    x=np.linspace(-np.pi/2,np.pi/2,ord+1)
    y=fun(x)
    
    # Fit a rational function 
    coeffs_rational=rational_fit(x,y,n,m)
    print(f"DEBUG: Rational coeffs numerator {coeffs_rational[:n]}")
    print(f"DEBUG: Rational coeffs denomenator {coeffs_rational[n:]}")
    # Fit a polynomial function
    coeffs_poly=polynomial_fit(x,y,ord)
    # Fit a cubic spline
    cubic_interp=interp1d(x,y,kind="cubic")
    
    # Sample finer resolution for evaluating the fits
    xfine=np.linspace(-np.pi/2,np.pi/2,1001)
    # Use rational fit to eval, and get y_true for baseline
    y_rational=rational(coeffs_rational,xfine,n,m)
    y_poly=polynomial(coeffs_poly,xfine)
    y_cubic_interp=cubic_interp(xfine)
    y_true=fun(xfine)
    
    ### Plots 
    plt.subplots(1,2,figsize=(8,4))
    # Plot data and fits
    plt.subplot(1,2,1)
    plt.plot(xfine,y_rational,label=f"best fit rational n={n},m={m}")
    plt.plot(xfine,y_poly,label=f"best fit polynomial, order={ord}")
    plt.plot(xfine,y_cubic_interp,label="cubic interpolation")
    plt.plot(xfine,y_true,"--",label="truth")
    plt.plot(x,y,"o",label="sample points")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel(f"{funcname}(x)")
    plt.title(f"{funcname}--fits and data")
    # Residuals
    plt.subplot(1,2,2)
    plt.plot(xfine,y_rational-y_true,label="y_rational-y_true")
    plt.plot(xfine,y_poly-y_true,label="y_poly-y_true")
    plt.plot(xfine,y_cubic_interp-y_true,label="y_cubic_interp-y_true")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Residuals")
    plt.title("Residuals")
    # Formatting stuff
    plt.tight_layout()
    plt.savefig(f"plots/fits4_laplacian_n={n}_m={m}.png")
    plt.show(block=True)
    


