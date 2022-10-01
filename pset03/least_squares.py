import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Load the data
dta=np.loadtxt("dish_zenith.txt",delimiter=" ",dtype=np.float64)
# Unpack data into 1d arrays
x,y,z=dta[:,0],dta[:,1],dta[:,2]

# Cast our data into the A matrix
A=np.vstack([np.ones(x.shape),x**2+y**2,x,y]).T
# Solve the chi-squared problem
m = inv(A.T@A)@A.T@z
print(f"INFO: Best fit params m={m}")

# Get the parameters from our fit
a = m[1]
x0 = m[2] / (-2*a)
y0 = m[3] / (-2*a)
z0 = m[0] - a*(x0**2 + y0**2)
print(f"INFO: Best fit params\na={a}\nx0={x0}\ny0={y0}\nz0={z0}")

# Lets plot the residuals, to see if we got something sensible
plt.figure(figsize=(6,4))
plt.plot(z - z0 - a*((x-x0)**2 + (y-y0)**2),"x")
plt.title("Least squares residuals\nchi-squared parabolic dish")
plt.savefig("img/least_squares_residuals.png",dpi=500)
plt.show()




