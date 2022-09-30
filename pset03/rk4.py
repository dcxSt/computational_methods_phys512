import numpy as np

def rk4_ks(f,x:float,y:float,h:float):
    """returns k1,k2,k3,k4"""
    k1=f(x,y)
    k2=f(x+0.5*h,y+h*k1*0.5)
    k3=f(x+0.5*h,y+h*k2*0.5)
    k4=f(x+h,y+h*k3)
    return k1,k2,k3,k4

def rk4_step(fun,x:float,y:float,h:float):
    k1,k2,k3,k4=rk4_ks(fun,x,y,h)
    return y + (k1+2*k2+2*k3+k4)*h/6

def rk4_stepd(fun,x:float,y:float,h:float):
    """Take a step of length h, compare that to two steps of length h/2
    fun : function
        dy/dx = fun(x,y)
    x : float 
        The x coordinates to time evolve
    y : float 
        The y coordinates to time evolve
    h : float
        The stepsize
    """
    # Take step h
    ynext=rk4_step(fun,x,y,h)
    ynext_hh=rk4_step(fun,x+0.5*h,rk4_step(fun,x,y,0.5*h),0.5*h)
    return (16*ynext_hh - ynext)/15


if __name__=="__main__":
    import matplotlib.pyplot as plt
    nsteps=200                      # number of steps
    x0,x1,y0=-20,20,1.0             # initial conditions
    x=np.linspace(x0,x1,nsteps+1)   # x values to evaluate on
    y=np.zeros(nsteps+1)            # initialize y array
    yhalf=np.zeros(nsteps+1)        # initialize y array for half step-size
    yopt=np.zeros(nsteps+1)         # initialize y array for half step-size
    y[0]=y0                         # initial y value
    yhalf[0]=y0                     # initial yhalf steps value
    yopt[0]=y0                      # initial y + (y-yhalf)/15 value
    h=np.median(np.diff(x))         # stepsize
    # Analytic solution
    ytrue=np.exp(np.arctan(x)) * y0/np.exp(np.arctan(x0))
    print("Integrating with rk4_step")
    def fun(x,y):
        return y/(1+x*x)
    for i in range(nsteps):
        y[i+1]=rk4_step(fun,x[i],y[i],h) # update y
        yh=rk4_step(fun,x[i],yhalf[i],0.5*h)         # update yhalf
        yhalf[i+1]=rk4_step(fun,x[i]+0.5*h,yh,0.5*h) # update yhalf
        yopt[i+1]=rk4_stepd(fun,x[i],yopt[i],h)     # update using rk4_stepd
        
    # Plot figure
    plt.subplots(2,1,figsize=(8,6))
    # Plot the numerically integrated function
    plt.subplot(211)
    plt.title("Integrated function")
    plt.plot(x,y,"-",label="integrated")
    plt.plot(x,yhalf,"-.",alpha=0.4,label="integrated half stepsize")
    plt.plot(x,yopt,"--",alpha=0.3,label="integrated combined")
    # plt.plot(x,ytrue,"--",alpha=0.4,label="analytic solution")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(which="both")
    # Plot the error
    plt.subplot(212)
    plt.title("Error, absolute value logplot")
    plt.semilogy(x,np.abs(ytrue-y),label="ytrue - y integrated")
    plt.semilogy(x,np.abs(ytrue-yhalf),label="ytrue - y integrated half stepsize")
    plt.semilogy(x,np.abs(ytrue-yopt),label="ytrue - y integrated combined")
    plt.xlabel("x")
    plt.ylabel("error in y")
    plt.legend()
    plt.grid(which="both")
    plt.tight_layout()
    plt.savefig("img/integrated_plots.png",dpi=500)
    plt.show(block=True)
    
    




