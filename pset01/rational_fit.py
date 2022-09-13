import numpy as np

# Evaluate x at the rational function with those coeffs
def rational(coeffs,x,n=n,m=m):
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
    invmat=np.linalg.pinv(mat)
    coeffs=invmat@y
    return coeffs

def polynomial_fit(x:np.ndarray, y:np.ndarray, ord:int):
    """Fit the data to a polynomial using linear algebra"""
    return 


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # The function we are fitting
    fun=np.cos
    funcname="cos" # This is for the plot
    
    # Rational function order of numer and denom polynomials
    n,m=3,3
    ord=n+m-1 # Oder of fit / order of polynomial fit
    
    # Set a modest number of points
    x=np.linspace(-np.pi/2,np.pi/2,ord)
    y=fun(x)
    
    # Fit a rational function 
    coeffs_rational=rational_fit(x,y,n,m)
    print(f"DEBUG: Rational coeffs numerator {coeffs_rational[:n]}")
    print(f"DEBUG: Rational coeffs denomenator {coeffs_rational[n:]}")
    
    # Sample finer resolution for evaluating the fits
    xfine=np.linspace(-np.pi/2,np.pi/2,1000)
    # Use rational fit to eval, and get y_true for baseline
    y_rational=rational(coeffs_rational,xfine,n,m)
    y_true=fun(xfine)
    
    ### Plots 
    plt.subplots(1,2,figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(x,y,"o",label="sample points")
    plt.plot(xfine,y_rational,label="best fit rational")
    plt.plot(xfine,ytrue,"--",label="truth")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel(f"{funcname}(x)")
    plt.title(f"{funcname}--fits and data")
    # Residuals
    plt.subplot(1,2,2)
    plt.plot(xfine,y_rational-ytrue,label="y_rational-ytrue")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Residuals")
    plt.title("Residuals")
    # Formatting stuff
    plt.tight_layout()
    plt.show(block=True)
    


