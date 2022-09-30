# Problem 1

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

The combination of the half steps that cancels the fifth order term is `(16*ynext_hh - ynext)/15`. This can be found in [Numerical Recipies](http://numerical.recipes/book/book.html). 

```python
def rk4_stepd(fun,x:float,y:float,h:float):
    """Take a step of length h, compare that to two steps of length h/2
    fun : function
        dy/dx = fun(x,y)
    x : float 
        The x coordinate
    y : float 
        The y coordinate 
    h : float
        The stepsize
    """
    # Take step h
    ynext=rk4_step(fun,x,y,h)
    ynext_hh=rk4_step(fun,x+0.5*h,rk4_step(fun,x,y,0.5*h),0.5*h)
    return (16*ynext_hh - ynext)/15
```

Since our `rk4` integrator requires 4 function evaluations, we need to make `4 + 2*4 = 12` in total. Four for `h`, then another eight for `h/2` for `h/2`.

**Bonus:** I used an RK4 integrator to solve [this](https://editor.p5js.org/dcxSt/sketches/VyBm8dgZ_) toy newtonian five body system planitary in P5 JS. You can play around with it by adding masses with `Mass(x,y,vx,vy,mass)` in the `setup` function.




