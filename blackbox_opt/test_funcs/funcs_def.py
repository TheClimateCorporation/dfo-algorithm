"""
Description:
This module contains the definitions of Ackley's, sphere,
Rosenbrock, Booth's, Bukin, Beale's functions. These functions are
called as blackbox functions from the run_test_func.py module.

These functions are popular global optimization test problems.

Functions and their optimal solution:

        -Rosenbrock: n-dimensional
            smooth, non-convex function. The minumum lies on a narrow flat
            valley. Convergence to global optimum is non-trivial.
            global optimum: f(x*) = 0, x* = all variables equal to 1

        - sphere: n-dimensional
            smooth function with no local minumum. Convex.
            global optimum: f(x*) = 0, x* = all variables equal to 0

        - Ackley's: 2-dimensional
            highly nonconvex-nonconcave, nonsmooth problem with many
            local solutions.
            global optimum: f(x*) = 0, x* = (0,0)

        - Beale: 2-dimensional
            non-convex, problem with multiple local solutions. Sharp corners.
            global optimum: f(x*) = 0, x* = (3,0.5)

        - Booth's: 2-dimensional
            smooth convex function.
            global optimum: f(x*) = 0, x* = (1,3)

        - Bukin: 2-dimensional
            non-smooth, nonconvex function with deep and steep valleys.
            global optimum: f(x*) = 0, x* = (-10,1)


input: 
    x: numpy.ndarray
        the input to each function
output:
    f(x): scalar
        the evaluation of the function at the given point x
author: Anahita Hassanzadeh
email: Anahita.Hassanzadeh@gmail.com
"""
import numpy as np
def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def sphere(x):
    """The sphere function"""
    res = sum(x**2)
    return res

def ackley(x):
    """The Ackley's function"""
    res = -20 * np.exp(-0.2 * np.sqrt(0.5 * sum(x**2))) - (
        - np.exp(0.5 * (sum(np.cos(2*np.pi*x))))) + np.e + 20
    return res

def beale(x_0):
    """The Beale function"""
    x = x_0[0]
    y = x_0[1]
    res = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (
        (2.625 - x + x*y**3)**2)
    return res

def booth(x_0):
    """The Booth function"""
    x = x_0[0]
    y = x_0[1]
    return ((x+2*y-7)**2 + (2*x+y-5)**2)

def bukin(x_0):
    """The Bukin function"""
    x = x_0[0]
    y = x_0[1]
    return (100 * np.sqrt(abs(y - 0.01*x**2)) + 0.01 * (abs(x+10)))
