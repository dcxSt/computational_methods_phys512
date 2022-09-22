import numpy as np
from numpy.polynomial.chebyshev import chebfit,chebval
log2=np.log2


"""Part a) Cheb fit"""
# Perpare data to fit
x=np.linspace(0.5,1,1001)
y=log2(x)
xscale=x*4-3 # Scale x between -1 and 1
# Fit a chebyshev polynomial
degree=7
cheb_coeffs=chebfit(xscale,y,deg=degree)
# Scale the fit so that it's in the right range
def fastlog2(x):
    return chebval(x*4-3,cheb_coeffs)

"""Test above code"""
# Make sure the error is okay by plotting residuals
xfine=np.linspace(0.5,1,10001)
res=fastlog2(xfine)-log2(xfine)

import matplotlib.pyplot as plt
plt.plot(xfine,res,label=f"fastlog2 - log2, degree={degree}")
plt.legend()
plt.xlabel("x")
plt.ylabel("log2(x)")
plt.title("Residuals of best fit chebyshev between 0.5 and 1")
plt.show(block=True)
    
"""Part a), get the log of any positive number."""
def mylog2(x):
    mantissa,exponant=np.frexp(x)
    # For positive numbers, the mantissa will never be less than 0.5
    return fastlog2(mantissa) + exponant

def myln(x):
    return np.log(2) * mylog2(x)

"""Test code above"""
x = np.linspace(0.1,100,100001)
plt.semilogy(x,(myln(x)-np.log(x)),label="Normalized residuals, (myln - np.log)/np.log")
plt.semilogy(x,myln(x),label="myln")
plt.semilogy(x,np.log(x),label="np.log")
plt.legend()
plt.xlabel("x")
plt.ylabel("ln x")
plt.title("Test of logs")
plt.tight_layout()
# plt.savefig("img/residuals_problem3.png")
plt.show(block=True)



