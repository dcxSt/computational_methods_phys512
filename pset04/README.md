# Pset4

*DISCLAIMER: Grades are stupid and the part of your job that involves assigning numbers to students is pointless—in fact, it is harmful because it's teaching kids to guess the teacher's password instead of ignighting their passion and enabling them to follow their curiosity. I worked as a grader for a few classes at McGill and found it cumbersome because students where comming and asking me about their grades all the time, and they're totally missing the point. Feedback is imporant, but not really in the form of a grade. I try very hard not to care about my grades because they stress me out and they invade my brain and block all the good thought pathways, so please give me a good one so that I can forget about it ASAP. Learning is not the activity of teaching, learning is the product of the activity of learners. If you have faith that I'm optimizing for learning and not for grades, please don't punish me for it by deducing silly little points for not presenting all of my calculations. The very notion of grades implies a bad-faith relationship between the student and the learning institution. They have it all upside-down at McGill. I know this because I've partaken in other kinds of learning experiances that work much better such as the [recurse center](https://www.recurse.com/about) and [school 2.0](https://school2point0.com/). If you liked this spiel and agree w/ me that the mainstream education system is structurally broken, you might like [Alfie Kohn](https://www.alfiekohn.org/blog/)'s writing. Thanks for being our TA. Your feedback is much appreciated.*

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

















