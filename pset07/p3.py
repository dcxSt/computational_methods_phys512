import numpy as np

def p(x):
    x[np.where(x<0)]=0.0
    return np.exp(-x)

def droplet(u):
    return -2*u*np.log(u)

def sample_exp(n:int):
    ymax=2/np.exp(1) # Height of accept region
    x=np.random.rand(n)      # Select random points in
    y=np.random.rand(n)*ymax # bounding box
    idxs=np.where(y<droplet(x)) # Accepted indices
    #plt.plot(x,y,".",markersize=1,label='rejected')
    plt.figure(figsize=(6,6*2/np.exp(1)))
    plt.plot(x,y,'.',markersize=1,label='rejected')
    plt.plot(x[idxs],y[idxs],".",markersize=1,label='accepted')
    plt.xlabel("u")
    plt.ylabel("v")
    plt.legend()
    plt.title("Samples accepted/rejected")
    plt.savefig("img/p3_sample_droplet.png",dpi=450)
    plt.show(block=True)
    return y[idxs]/x[idxs]


import matplotlib.pyplot as plt
n=100000
u=np.linspace(0,1,n)
v=droplet(u)
plt.figure(figsize=(6,6*2/np.exp(1)))
plt.fill_between([0,1],[2/np.exp(1)]*2,alpha=0.3)
plt.fill_between(u,v,alpha=0.5)
plt.title("Acceptance Region, Exponential ROU")
plt.savefig("img/p3_rou_acceptance_region.png",dpi=450)
plt.xlabel("u")
plt.ylabel("v")
plt.show(block=True)

samples=sample_exp(n)
samples=samples[np.where(samples<10)]
plt.hist(samples,bins=50,density=True,label="ROU histogram")
x=np.linspace(0,10,1000)
plt.plot(x,np.exp(-x),label="y=e^-x")
plt.title("ROU Exponential")
plt.xlabel("x")
plt.ylabel("PDF esimate / dN/dt")
plt.savefig("img/rou_exponential.png",dpi=450)
plt.show(block=True)








