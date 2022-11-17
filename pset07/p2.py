import numpy as np

def sample_power_law(n,alpha):
    # Sample s^-alpha
    y=np.random.rand(n)
    # power law transformation
    x=y**(1/(1-alpha))# inverse CDF
    return x

def sample_reject(g,f,sample_f,n:int):
    """
    Sample n from f's dist (sample_f). Compute random height,
    accept those that land under the g curve.
    """
    # sample f 
    f_samp=sample_f(n)
    # reject some of these
    y_samp=np.random.rand(len(f_samp))*f(f_samp)
    g_select=f_samp[np.where(y_samp<=g(f_samp))]
    g_reject=f_samp[np.where(y_samp> g(f_samp))]
    return g_select,g_reject,y_samp

def rejection_method(g,f,sample_f,n):
    """
    f bounds g from above. 
    We sample f and reject some of the sampled values.

    n : int
        number of samples returned
    """
    g_samp=np.array([]) # init empty array
    while len(g_samp)<n:
        n_over=int((n-len(g_samp))*1.2) # over sample to terminate
        g_select,_,_ = sample_reject(g,f,sample_f,n_over)
        g_samp=np.hstack([g_samp,g_select])
    return g_samp[:n]




# PLot the power law
import matplotlib.pyplot as plt
alpha=2
n=50000
#x=sample_power_law(n,alpha)
#x=x[np.where(x<30)]# throw out tail
#plt.figure(figsize=(6,6))
#plt.clf()
#plt.hist(x,bins=50,density=True,label="data from transformed random number generator")
#plt.plot(np.linspace(1,30,50),np.linspace(1,30,50)**(-alpha),label="Power law function")
#plt.legend()
#plt.title("Power law")
#plt.xlabel("s")
#plt.ylabel("Relative Density")
#plt.yticks([])
#plt.savefig("img/p2_plaw.png",dpi=450)
#plt.show(block=True)


arrin=np.linspace(1,100,500)
#ypow=arrin**(-alpha)
#yexp=np.exp(-arrin)
#plt.loglog(arrin,ypow,label=f"power {alpha}")
#plt.loglog(arrin,yexp,label="exponential")
##plt.loglog(arrin,ypow/yexp,label=f"ratio")
##plt.loglog(arrin,np.ones(arrin.shape))
#plt.legend()
#plt.title("powerlaw bounding exponential")
#plt.savefig("img/p2_powerlaw_bound_exp.png")
#plt.show(block=True)

# Over-sample from f to get g sampler
# f is a power law
beta=2.1
alpha=1.7
f=lambda xin:xin**(-alpha)/beta
sample_f=lambda n:sample_power_law(n,alpha)
# g is an exponential decay
g=lambda t:np.exp(-t)
xin=np.linspace(1,10,100)

f_samp=sample_f(n) # sample from f
f_samp=f_samp[np.where(f_samp<=10)] # trunkate
height=np.random.rand(len(f_samp))*f(f_samp) # get random heights
idx_under_g=np.where(height<=g(f_samp)) # idxs of heights under g
g_samp=f_samp[idx_under_g]
plt.figure(figsize=(6,6))
plt.plot(f_samp,height,".",markersize=.2,label="rejected")
plt.plot(f_samp[idx_under_g],height[idx_under_g],".",markersize=.2,label="accepted")
plt.plot(xin,f(xin),linewidth=0.4,label="powerlaw")
plt.plot(xin,g(xin),label="exponential")
plt.legend()
plt.title(f"Rejection sampling, powerlaw alpha={alpha}, exponential")
#plt.savefig("img/p2_accepted_rejected.png")
plt.savefig("img/p2_accepted_rejected_skinny.png")
plt.show(block=True)

# Now we plot a histogram and compare it to the curve
g_samp=f_samp[idx_under_g]
plt.figure(figsize=(6,6))
plt.hist(g_samp,density=True,bins=50,label="sampled points")
plt.plot(xin,g(xin)/g(1),label="expected exponential")
plt.yticks([])
plt.ylabel("PDF / sample denisty")
plt.xlabel("s")
plt.legend()
plt.title("Exponential Rejection Sampling")
#plt.savefig("img/p2_hist.png")
plt.savefig("img/p2_hist_skinny.png")
plt.show(block=True)







