"""
Original Python code extracted from arXiv:2301.05284 (Appendix B)
Authors: K.A. Katalova (Dragunova), N. Nikbakht, I.D. Remizov (2023)
"Concrete examples of the rate of convergence of Chernoff approximations:
 numerical results for the heat semigroup"

Environment: Python 3.8.3, Jupyter Notebook 6.1.4, Anaconda 3
Transcribed from screenshots (codd1.png - codd4.png)
"""

# importing moduli
import matplotlib.pyplot as plt
from sympy import oo
from scipy import integrate
import numpy as np
import math

# variables declaration
tau = 1/2
n = 10

l = []
for i in range(1, n + 1):
    l.append(math.log(i))


# almost self-contained class
class classoffunctions(object):
    def thefunction(self, x):
        return x


# Chernoff function (S(t)f)(x) = (2/3)f(x) + (1/6)f(x + (6t)^(1/2)) + (1/6)f(x - (6t)^(1/2))
# Second-order tangency to the Laplacian (k=2)
def oper(inputobject: classoffunctions, t):
    outputobject = inputobject
    f = inputobject.thefunction
    def funct(x):
        return (2/3)*f(x) + (1/6)*f(x + (6*t)**(1/2)) + (1/6)*f(x - (6*t)**(1/2))
    outputobject.thefunction = funct
    return outputobject


# Chernoff function (G(t)f)(x) = (1/4)*f(x+2t^(1/2)) + (1/4)*f(x-2t^(1/2)) + (1/2)*f(x)
# First-order tangency to the Laplacian (k=1)
def oper1(inputobject: classoffunctions, t):
    outputobject = inputobject
    f = inputobject.thefunction
    def funct(x):
        return (1/4)*f(x + 2*t**(1/2)) + (1/4)*f(x - 2*t**(1/2)) + (1/2)*f(x)
    outputobject.thefunction = funct
    return outputobject


# composition degree of Chernoff function (S(t)f)(x) — n-fold composition C(tau/n)^n
def degr(g, tau, n):
    y = []
    obj = classoffunctions()
    for n_p in range(1, n + 1):
        obj_k = obj
        obj_k.thefunction = g
        for k in range(1, n_p + 1):
            obj_k = oper(obj_k, tau / n_p)
        y.append(obj_k.thefunction)
    return y


# composition degree of Chernoff function (G(t)f)(x)
def degr1(g, tau, n):
    y = []
    obj = classoffunctions()
    for n_p in range(1, n + 1):
        obj_k = obj
        obj_k.thefunction = g
        for k in range(1, n_p + 1):
            obj_k = oper1(obj_k, tau / n_p)
        y.append(obj_k.thefunction)
    return y


# norm computation
def norm(y, sol):
    d = []
    for n_p in range(0, n):
        d.append(np.max(np.abs(sol - y[n_p](x))))
    return d
