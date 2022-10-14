# Pset4


*DISCLAIMER: Grades are stupid and the part of your job that involves assigning numbers to students is pointless—in fact, it is harmful because it's teaching kids to guess the teacher's password instead of ignighting their passion and enabling them to follow their curiosity. I worked as a grader for a few classes at McGill and found it cumbersome because students where comming and asking me about their grades all the time, and they're totally missing the point. Feedback is imporant, but not really in the form of a grade. I try very hard not to care about my grades because they stress me out and they invade my brain and block all the good thought pathways, so please give me a good one so that I can forget about it ASAP. Learning is not the activity of teaching, learning is the product of the activity of learners. If you have faith that I'm optimizing for learning and not for grades, please don't punish me for it by deducing silly little points for not presenting all of my calculations. The very notion of grades implies a bad-faith relationship between the student and the learning institution. They have it all upside-down at McGill. I know this because I've partaken in other kinds of learning experiances that work much better such as the [recurse center](https://www.recurse.com/about) and [school 2.0](https://school2point0.com/). If you liked this spiel and agree w/ me that the mainstream education system is structurally broken, you might like [Alfie Kohn](https://www.alfiekohn.org/blog/)'s writing. Thanks for being our TA. Your feedback is much appreciated.*

*All code can be found in `./mcmc/problem1.py`. I made an effort to document it sufficiently, and keep it fairly organized.*

## Problem 1 (a)

*The code for this problem can be found in the file with this relative path `./mcmc/a.py`*

Using the standard notation `A(m)=d` where `m=(a,t0,w)` is the three tuple of our model parameters, we cast this into a non-linear chi square optimization. 

```python
chisq=(A(m)-d).T@Ninv@(A(m)-d)
gradchisq=-2*gradA.T@Ninv@(A(m)-d)
```

We will approximate the curvature using only first derivatives of our non-linear model `A`. 

```python
def A(m,t):
    """assumes m is the three vector m=(a,t0,w), or matrix with 3 cols"""
    a,t0,w=m.T
    return a/(1+((t-t0)/w)**2)

def gradA(m,t):
    """assumes m is the three vector m=(a,t0,w), or mat w/ three cols"""
    a,t0,w=m.T
    dAda=1/(1+((t-t0)/w)**2)
    dAdt0=2*a*(t-t0)/(w**2*(1+((t-t0)/w)**2)**2)
    dAdw=2*a*(t-t0)**2/(w**3*(1+((t-to)/w)**2)**2)
    return np.vstack([dAda,dAdt,dAdw]).T
```

Now, assuming the most basic noise model (uncorrelated uniform gaussian noise, i.e. `Ninv` is the identity matrix), we can write a simple newton iteragor

```python
from numpy.linalg import inv
def newton_iter(m,t,d):
    """Returns next iteration of newton's method"""
    r=d-A(m,t) # residuals
    Ap=gradA(m,t)
    return m + inv(Ap.T@Ap)@Ap.T@r 
```

Now we call this function a few times and find that it converges very quickly given sensible starting parameters. 

```python
dta=np.load("sidebands.npz")
t=dta['time']
d=dta['signal']
print("INFO: Finding best fit parameters")
# initiate m with some sensible parameters
w=1.0e-5
t0=np.mean(t)
a=np.max(d)
m0=np.array((a,t0,w))  
m=m0.copy()
for i in range(10):
    m=newton_iter(m,t,d)
```

The optimal parameters are found to be

```
a  = 1.423
t0 = 1.924e-04
w  = 1.792e-05
```

Plot of the RMS of the residuals converging very quicklly which justifies the number of iterations, and plot of the model and the residuals.

![q1a_best_fit](https://user-images.githubusercontent.com/21654151/195674173-8eaf7aea-f1d0-46fe-809f-c55949f7cc1f.png)
![q1a_residuals](https://user-images.githubusercontent.com/21654151/195674177-29c3aaff-43f9-47f7-ad7e-f0db3e7e6d4c.png)
![q1a_newton_converge](https://user-images.githubusercontent.com/21654151/195674175-221ae87b-9143-4b8f-9f79-a3c3cd47c3bf.png)

We conclude that the model is not perfect, because it (a) doesn't account for the sidelobes, and (b) doesn't account for the little kink at the peak of our data. 


## Problem 1 (b)

Estimate the noise in your data and use that to estimate the errors in your parameters. 

The noise is about `sigma=np.mean(abs(d-A(m,t)))`. Assuming uncorrelated noise, our inverse noise matrix can be estimated as an identity matrix divided by sigma squared `Ninv=1/sigma**2`. Reading off the diagonals of `inv(Ap.T@Ninv@Ap)`, we get appropriate estimates of the errors in our parameters. 

```
sigma_a  = 3.26e-04
sigma_t0 = 4.11e-09
sigma_w  = 5.82e-09
```

It is instructive for intuition to look at appropriately scaled values

```
sigma_a/a         = 2.29e-04
sigma_t0/(tf-ti)  = 1.03e-05
sigma_w/w         = 3.25e-04
```

This is good, I expected them to all be about the same order of magnitude. 

## Problem a (c)

Repeat part (a) but use numerical derivatives. 

First we ballpark the optimal stepsize for a 2-point central numerical derivative by taylor expanding, throwing out higher order terms, adding the roundoff error divided by dx as a random variable, and taking a derivative of what's left to minimize this one, assuming the third derivative `f'''(x)` is order 1. We find that `dx=2np.sqrt(epsilon)`, where epsilon is the roundoff error, which we approximate as `1.0e-16`. So we set our stepsize to `dx=2.0e-08` (aka `h`). 

Now we write a numerical partial derivative operator

```python
def ndiff(f,x,idx):
    """Take the partial derivative of f at x wrt to the idx'th argument.

    f : function
        A multivariate function
    x : np.ndarray
        A vector argument, where we evaluate the derivaitve at
    idx : int
        The index of the direction along which we evaluate the partial 
        deriv
    """
    # Select some optimal step sizes
    # We approximate roundoff error epsilon=10.0e-16
    # Highest order term in 2 point derivative TS expansion is o(dx**3)
    # Optimal step size is about 2*sqrt(epsilon) = 2.0e-08
    # (ballpark back of envolope estimate)
    dx = 2.0e-08
    step_h = np.zeros(x.shape)
    step_h[idx] = dx
    # Return the numerical partial derivative of f wrt it's argument at 
    # idx, at x
    return (f(x+step_h) - f(x-step_h))/(2*dx)
```

Then we use this partial derivative operator to write a numerical gradient opertor for our function `A`. Noting that python syntax is so lovely that we can just evaluate this with a vector `t`. 

```python
def numerical_gradA(m,t):
    """Numerically compute the gradiant of A wrt m at m,t"""
    dAda  = ndiff(lambda m:A(m,t),m,0)
    dAdt0 = ndiff(lambda m:A(m,t),m,1)
    dAdw  = ndiff(lambda m:A(m,t),m,2)
    return np.vstack([dAda, dAdt0, dAdw]).T
```

Now we adapt our newton iteration to use our numerical derivative. 

```python
def newton_iter_numerical(m,t,d):
    """Returns next iteration of newton's method"""
    r=d-A(m,t) # residuals
    Ap=numerical_gradA(m,t)
    return m + inv(Ap.T@Ap)@Ap.T@r # Trivial Ninv's cancel
```

Running this we find that our best fit parameters agree extreemely well with the analytical derivatives. The answers are not statistically different since they are well within the estimated std in the noise of each other. Below the subscripts `numerical` and `analytic` indicate which method was used to evaluate the derivatives. We compare these differences with the estimated errors in our parameters to great satisfaction. 

```
a_numerical-a_analytic   = 6.20e-09
err_a                    = 3.26e-04

t0_numerical-t0_analytic = -4.53e-13
err_t0                   = 4.11e-09

w_numerical-w_analytic   = -1.56e-13
err_w                    = 5.82e-09
```

## Problem 1 (d) 

We adapt our code above to deal with a the more general case. The numerical derivative operator remains the same, but the gradient and newton iterator are generalized and updated. 

```python
def numerical_grad(A,m,t):
    """Numerically compute the gradiant of A wrt m at m,t

    A : function
        Assumes A takes two arguments (params, times)
    m : array-like
    t : np.ndarray
    """
    return np.vstack([ndiff(lambda m:A(m,t),m,idx) for idx in range(len(m))]).T

def newton_iter_numerical(A,m,t,d):
    """Returns next iteration of newton's method

    A : function / model
        Takes two params (model_params, times)
    m : array-like
        our model parameters, passed to A as first arg
    t : np.ndarray
        Times at which LASER beam is measured
    d : measured data
    """
    r=d-A(m,t) # residuals
    Ap=numerical_grad(A,m,t)
    return m + inv(Ap.T@Ap)@Ap.T@r # Ninv's cancel
```

Plots of this new fit. 


![q1d_best_fit](https://user-images.githubusercontent.com/21654151/195946278-d080402d-18de-45f1-b2cf-77d2821b1b7b.png)

![q1d_residuals](https://user-images.githubusercontent.com/21654151/195946291-7ed4aca9-db98-4f62-9694-5f0f23f173cf.png)

![q1d_newton_converge](https://user-images.githubusercontent.com/21654151/195946301-d5caf908-cf4c-448a-8fc2-d519783efc90.png)


The estimated errors in the measurement are

```
sig_a  = 2.30e-04
sig_b  = 2.20e-04 
sig_c  = 2.15e-04 
sig_t0 = 2.73e-09 
sig_dt = 3.29e-08 
sig_w  = 3.39e-17
```









