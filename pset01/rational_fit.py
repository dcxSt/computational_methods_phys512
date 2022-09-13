import numpy as np
import matplotlib.pyplot as plt

# Define some constants
pi=np.pi

# Rational function fit hyper-parameters
n=3
m=3
# The function we are fitting
fun=np.cos
funcname="cos" # This is for the plot

# def rational(coeffs,x,n=n,m=m):
#     """x is a float"""
#     numerator = np.dot(coeffs[:n],np.array([x**i for i in range(n)]))
#     denom = 1 + np.dot(coeffs[n:],np.array([x**i for i in range(1,m)]))
#     return numerator/denom


# Evaluate x at the rational function with those coeffs
def rational(coeffs,x,n=n,m=m):
    """Evaluates the rational function p(x)/(1+qq(x))

    Does the same thing as rational ^^ above commented, but can take
    vector intput x. 

    coeffs : ndarray
        The coefficients of p(x) and qq(x)  

    x : ndarray or float/int
        What we are evaluating.

    n : int
        The number of numerator coeffs

    m : int
        The number of denom, non-constant coeffs
    """
    assert coeffs.shape==(n+m-1,) # Sanity check
    num_coeffs = coeffs[:n]
    denom_coeffs = np.hstack([[1],coeffs[n:]])
    numerator = np.vstack([x**i for i in range(n)]).T@num_coeffs
    denom = np.vstack([x**i for i in range(m)]).T@denom_coeffs
    return numerator/denom


# Set a modest number of points
x=np.linspace(-pi/2,pi/2,n+m-1)
y=fun(x)
# Finer resolution for evaluating the fit
xfine=np.linspace(-pi/2,pi/2,1000)


### Compute the coefficients
# First, setup the coeffs matrix we need to invert
# NB, even though we use for loops, we iterate over small ranges
# The part of the matrix corresponding to p(x)
mat_p=np.zeros((n+m-1,n))
for k in range(n):
    mat_p[:,k] = x**k

# The part of the matrix corresponding to qq(x)
mat_qq=np.zeros((n+m-1,m-1))
for k in range(1,m):
    mat_qq[:,k-1] = -y * (x**k)

# Stack the 'componant' matrices to get a square matrix
mat=np.hstack((mat_p,mat_qq))

# Invert the matrices, solve for the coefficients
invmat=np.linalg.pinv(mat)
coeffs=invmat@y

print(f"INFO: coeffs_p={coeffs[:n]}\ncoeffs_qq={coeffs[n:]}")

# Use rational fit to eval, and get y_true for baseline
y_rational=rational(coeffs,xfine)
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




