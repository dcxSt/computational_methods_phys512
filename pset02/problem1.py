"""
One can work out the electric field from an infinitessimally thin spherical shell of charge with radius R by working out the field from a ring along its central axis, and integrating those rings to form a spherical shell. Use both your integrator and scipy.integrate.quad to plot the electric field from the shell as a function of distance from the center of the sphere. Make sure the range of your plot covers regions with z < R and z > R. Make sure one of your z values is R. Is there a singularity in the integral? Does quad care? Does your integrator? Note - if you get stuck setting up the problem, you may be able to find solutions to Griffiths problem 2.7, which sets up the integral.
"""

import numpy as np
from scipy.integrate import quad
pi=np.pi
sqrt=np.sqrt
cos=np.cos
arctan=np.arctan


def integrate(fun,a,b,tol,rec_level=0):
    """Integrates fun over the interval [a,b] with adaptive 5-point integral"""
    x=np.linspace(a,b,5)
    dx=x[1]-x[0]
    y=fun(x)
    i1=(y[0]+4*y[2]+y[4])/3*(2*dx)
    i2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3*dx
    myerr=np.abs(i1-i2)
    if myerr<tol:
        return i2
    elif rec_level>=10:
        print("WARNING: recursion level 10 has been reached, stopping here.")
        print(f"WARNiNG: Error is still too large, myerr={myerr}")
        return i2
    else:
        # Reccusion step
        mid=(a+b)/2
        int1=integrate(fun,a,mid,tol/2,rec_level+1)
        int2=integrate(fun,mid,b,tol/2,rec_level+1)
        return int1+int2


"""main"""
# Set up the integral
R=1      #Radius of the sphere, (charge density is assumed to be 1)
tol=0.01 #Error Tolerance
xarr=np.arange(0,8,1/4) #Distances from center 0 to evaluate integral at
result_integrate=[] #Stores the values returned by our integrator
result_quad=[]      #Stores the values returned by scipy's integrator 
# Integrate over y for each distance that we care about
for x in xarr:
    def fun(y,R=R,x=x): 
        return 2*pi*sqrt(R**2-(x-y)**2)/(R**2+y**2)*cos(arctan(sqrt(R**2-(x-y)**2)/y))
    a,b=x-R,x+R
    print(f"DEBUG: x{x}, R{R}, a{a}, b{b}")
    #Apply Custom integrator
    result_integrate.append(integrate(fun,a,b,tol))
    #Apply Scipy's Fortran 'quadrature' integrator
    result_quad.append(quad(fun,a,b)[0])


import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))

plt.plot(xarr,result_quad,label="Scipy.integrate.quad")
plt.plot(xarr,result_integrate,"--",label=f"Custom integrate, tolerance={tol}")
plt.xlabel("Distance from center")
plt.ylabel("Electric field")
plt.title(f"Numerical Integration uniformly charged sphere R={R}")
plt.legend()
plt.tight_layout()
plt.savefig("img/nintegrate.png")

plt.show()



    




