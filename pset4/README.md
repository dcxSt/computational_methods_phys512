# Pset4

*DISCLAIMER: Grades are stupid and your job as grader is pointless—in fact, it is harmful because it's teaching kids to guess the teacher's password instead of ignighting their passion and enabling them to follow their curiosity. Feedback is imporant, but not really in the form of a grade. I try very hard not to care about my grades because they stress me out and they invade my brain and block all the good thought pathways, so please give me a good one so that I can forget about it ASAP. Learning is not the activity of teaching, learning is the product of the activity of learners. If you have faith that I'm optimizing for learning and not for grades, please don't punish me for it by deducing silly little points for not presenting all of my calculations. The very notion of grades implies a bad-faith relationship between the student and the learning institution. They have it all upside-down at McGill. I know this because I've partaken in other kinds of learning experiances that work much better such as the [recurse center](https://www.recurse.com/about) and [school 2.0](https://school2point0.com/). If you liked this spiel and think our education system is structurally broken, you might like [Alfie Kohn](https://www.alfiekohn.org/blog/)'s writing. Thanks for being our TA. Your feedback is much appreciated.*

## Problem 1 (a)

Using the standard notation `A(m)=d` where `m=(a,t0,w)` is the three tuple of our model parameters, we cast this into a non-linear chi square optimization. 

```python
chisq=(A(m)-d).T@Ninv@(A(m)-d)
gradchisq=-2*gradA.T@Ninv@(A(m)-d)
```

We will approximate the curvature using only first derivatives of our non-linear model `A`. 

```python

```



