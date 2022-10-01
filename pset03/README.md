# Problem 1

The code for this problem is in `rk4.py`

*Write an RK4 integrator with prototype to take one step:*

```python
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
```

*Now write another stepper `rk4_stepd` that takes a step of length `h`, compares that to two steps of length h/2, and uses them to cancel out the leading-order error term from RK4. How many function evaluations per step does this one use? Use this modified stepper to carry out the same ODE solution using the same number of function evaluations as the original. Which is more accurate?*

The combination of the half steps that cancels the fifth order term is `(16*ynext_hh - ynext)/15`. This can be found in [Numerical Recipies](http://numerical.recipes/book/book.html). We don't derive this result because it would litterally involve taking like 50 derivatives. 

```python
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
    # Take one step of size h
    ynext=rk4_step(fun,x,y,h)
    # Take two steps of size 0.5*h
    ynext_half=rk4_step(fun,x,y,0.5*h)
    ynext_2half=rk4_step(fun,x+0.5*h,ynext_half,0.5*h)
    # Return the combined formula to cancel order h**5 terms
    return (16*ynext_2half - ynext)/15 
```

Since our `rk4` integrator requires 4 function evaluations, we need to make `4 + 2*4 = 12` in total. Four for `h`, then another eight for `h/2` for `h/2`.

![integrated_plots](https://user-images.githubusercontent.com/21654151/193368230-8837df3e-c98d-4dca-ae87-6e117dd9dd69.png)

**Bonus:** I used an RK4 integrator to solve [this](https://editor.p5js.org/dcxSt/sketches/VyBm8dgZ_) toy newtonian n-body planitary system in P5 JS. You can play around with it by initiating mass objects with `Mass(x,y,vx,vy,mass)` in the `setup` function!


# Problem 2

The code for this problem is in `u238decay.py`

*(a) Write a program to solve for the decay products of U238 (refer to slides for the decay chain). You can use the ODE solver from scipy, but you’ll need to set the problem up properly. Please make sure to include all the decay prodcuts in the chain. Assume you start from a sample of pure U238 (in nature, this sort of separation happens chemically when rocks are formed). Which solver would you use for this problem?*

As discussed in class, we don't want to turn down the stepsize too small, so we use the Radau integrator, instead of your run-of-the-mill Runge-Kutta. In our update function, we take advantage of the fact that our final product is stable, to use the `np.roll` function. To implement this we use `np.inf`, which sends floats to zero when we divide by that. 

```python
# Define time variables in seconds to help the eye
minu=60
hour=60*minu
day=24*hour
year=365.25*day

# Half life of ordered products in the decay chain
halflife=np.array([
    4.468*1e9*year, # U238 has a long halflife, idx=0
    24.10*day,
    6.70*hour,
    245500*year, # U234, idx=3
    75380*year, # Th230, idx=4
    1600*year,
    3.8235*day,
    3.10*minu,
    26.8*minu,
    19.9*minu,
    164.3*1e-6, # pb 204 is very small, idx=10
    22.3*year,
    5.015*year,
    138.376*day,
    np.inf # pb 206 is stable, idx=14 (or -1 in python)
    ])

# Dictionary of important indices for easy access
idx = {"U238":0,"U234":3,"Th230":4,"Pb204":10,"Pb206":14}

#tau=np.log(2)/halflife # Tau time constant, diagonal matrix
decay_rate=np.log(2)/halflife
```

Our update function reads

```python
# we can use roll because final product is stable, this is very sneaky, i know
def decay_timestep(x,y,decay_rate=decay_rate):
    """dy/dx = decay_timestep(x,y)"""
    return -y*decay_rate + np.roll(y*decay_rate,1) 
```

Now we can integrate

```python
from scipy.integrate import solve_ivp
x0,x1=0,halflife[idx["U238"]]
y0=np.zeros(decay_rate.shape)
y0[0]=1.0
ans=solve_ivp(decay_timestep,[x0,x1],y0,method='Radau',t_eval=np.linspace(x0,x1,1000))
t=ans['t']
y=ans['y']
u238=y[idx["U238"]]
pb206=y[idx["Pb206"]]
```

*(b) Plot the ratio of Pb206 to U238 as a function of time over a region where it’s interesting. Does this make sense analytically? (If you look at the decay chain, all the half-lives are short compared to U238, so you can approximate the U238 decaying instantly to lead.*

Yes, it makes sense, for the reason provided in the question! It makes sense intuitively because if we ignore all the intermediate products, we expect exponential decay of uranium-238 and exponential growth of pb230, which is approximately what the plots show us. 

![u238_pb206](https://user-images.githubusercontent.com/21654151/193378458-8bb3cb35-cd0e-42c3-a835-bc5a2249b627.png)

*Now plot the ratio of Thorium 230 to U234 over a region where that is interesting. Radioactive decay is frequently used to date rocks, and these results point at how you can determine the age of a uranium-bearing rock that is anywhere from thousands to billions of years old. (Of course, in this case the starting ratio of U234 to U238 would probably have already reached its long-term average when the rock was formed, but you could still use the U234/Th230 ratio under that assumption.)*

The halflife of Thorium 230 is order of a hundred thousand years, 

```python
x0=0
x1=30*halflife[idx["Th230"]]
y0=np.zeros(decay_rate.shape)
y0[0]=1.0 # start with just uranium
ans=solve_ivp(decay_timestep,[x0,x1],y0,method='Radau',t_eval=np.linspace(x0,x1,1000*100))
t=ans['t']
y=ans['y']
u238=y[idx["U238"]]
th230=y[idx["Th230"]]
```

After a bit of experimentation, it takes a few halflives for Thorium230 to reach equilibrium. Since decay rate is proportional to quanity, it makes sense that the quantity starts growing, proportional to the decay rate of Uranium-238, and then slows to an equilibrium value as the decay rate of Th230 grows to balance it's rate of creation. 

















