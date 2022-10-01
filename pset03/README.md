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

![u238_th230](https://user-images.githubusercontent.com/21654151/193379040-9f37a177-16fb-4ba8-bc25-3c197752c259.png)

# Problem 3

The code for this problem is in `least_squares.py`

*We’ll do a linear least-squares fit to some real data in this prob- lem. Look at the file dish zenith.txt. This contains photogrammetry data for a prototype telescope dish. Photogrammetry attempts to reconstruct surfaces by working out the 3-dimensional positions of targets from many pictures (as an aside, the algorithms behind photogrammetry are another fun least-squares- type problem, but beyond the scope of this class). The end result is that dish zenith.txt contains the (x,y,z) positions in mm of a few hundred targets placed on the dish. The ideal telescope dish should be a rotationally symmetric paraboloid. We will try to measure the shape of that paraboloid, and see how well we did.*

*(a) Helpfully, I have oriented the points in the file so that the dish is pointing in the +z direction (in the general problem, you would have to fit for direction the dish is pointing in as well, but we will skip that here). For a rotationally symmetric paraboloid, we know that*

```python
z - z0 = a*((x - x0)**2 + (y - y0)**2)
```

*and we need to solve for x0 , y0 , z0 , and a. While at first glance this problem may appear non-linear, show that we can pick a new set of parameters that make the problem linear. What are these new parameters, and how do they relate to the old ones?*

Lets factor this formula, and create new parameters, `mu0` through `mu3`. We need four of them since we have four parameters to start with. 

```python
z = z0 + a*(x0**2 + y0**2) + a*(x**2 + y**2) + a*(x**2 + y**2) - 2*a*x0*x - 2*a*y0*y
mu0 = z0 + a*(x0**2 + y0**2)
mu1 = a
mu2 = -2*a*x0
mu3 = -2*a*y0
```

With these new parameters, our equations are linear. To cast it into the regular form `A@m=d`, let `d=z`, where `z` is a k-vector, `m` is a horizontal stack of 4xk columns of our new variables `mu0,mu1,mu2,mu3`, and `A` is an kx4 vertical stack of rows with data `1,x**2+y**2,x,y`. 

*(b) Carry out the fit. What are your best-fit parameters?*

Now, we carry out the fit assuming uncorrelated, uniform noise. This makes our lives simple because we can just throw out the `N` matrix. 

```python
chisq = (d-A@m).T@(d-A@m)
```

Taking a derivative and then setting the result to zero, we get

```python
A.T@A@m == A.T@d
```

Since `A.T@A` is just a four by four matrix, we can invert it. 

```python
m = np.inv(A.T@A)@A.T@d
```

The code to load the data and carry out a least squares fit is displayed here

```python
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
```

The best fit parameters obtained are

```python
a=0.00016670445477401277
x0=-1.3604886221973425
y0=58.22147608157945
z0=-1512.8772100367855
```

The residuals look like this 

```python
plt.plot(z - z0 - a*((x-x0)**2 + (y-y0)**2),"x")
```

![least_squares_residuals](https://user-images.githubusercontent.com/21654151/193386423-ab54cbca-6947-43b4-bd42-c88585e31d9c.png)

Unfortunately these resituals look pretty correlated, so our assumption that the noise was random wasn't the best. 

*(c) Estimate the noise in the data, and from that, estimate the uncertainty in a. Our target focal length was 1.5 metres. What did we actually get, and what is the error bar? In case all facets of conic sections are not at your immediate recall, a parabola that goes through (0,0) can be written as y = x2/(4f) where f is the focal length. When calculating the error bar for the focal length, feel free to approximate using a first-order Taylor expansion.*

The uncertainty in `a` can be estamted by 

```python
# Estimate noise
noise_mat = np.diag(z - A@m)
cov_mat = inv(A.T@inv(noise_mat)@A)
sigma_a = np.sqrt(cov_mat[1,1])
print(f"INFO: Uncertainty in a = {delta_a}")
```

This gives an uncertainty in a of `2.66e-08`. The focal length which is `f=1/(4*a)` is computed to be `1499.66` millimeters. The error on this can be approximated with taylor expansion. So `sigma_f` is the absolute value of the derivative of `f` wrt `a`, times `sigma_a`. This gives 

```python
sigma_f = sigma_a / (2 * a**2)
```

Which gives a `sigma_f` of `3.989` millimeters. 




























