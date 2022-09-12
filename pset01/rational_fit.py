import numpy as np
import matplotlib.pyplot as plt

n=3
m=3
fun=np.sin

def rational(coeffs,x,n=n,m=m):
    """x is a float"""
    numerator = np.dot(coeffs[:n],np.array([x**i for i in range(n)]))
    denom = 1 + np.dot(coeffs[n:],np.array([x**i for i in range(1,m)]))
    return numerator/denom

x=np.linspace(0,3,n+m-1)
y=fun(x)

matleft=np.zeros((n+m-1,n))
for rowidx in range(n+m-1):
    for entry in range(n):
        matleft[rowidx,entry] = x[rowidx]**entry

matright=np.zeros((n+m-1,m-1))
for rowidx in range(n+m-1):
    for entry in range(1,m):
        matright[rowidx,entry-1] = -y[rowidx]*x[rowidx]**entry

mat=np.hstack((matleft,matright))
# print(mat.shape)
# print(mat)

invmat=np.linalg.pinv(mat)

coeffs=invmat@y
print(f"INFO: hurray we got coeffs {coeffs}")
print(rational(coeffs,0.0))

xfine=np.linspace(0,3,1000)
yfine=np.array([rational(coeffs,i) for i in xfine])
ytrue=fun(xfine)

plt.figure()
plt.plot(x,y,"o",label="sample points")
plt.plot(xfine,yfine,label="best-fit")
plt.plot(xfine,ytrue,"--",label="truth")
plt.legend()
# plt.plot(x,rational(coeffs,x))
plt.show(block=True)




