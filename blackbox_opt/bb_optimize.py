"""
Description: This module calls five derivative free algorithms
from the scipy.optimize library as well as the DFO algorithm whose
source code is included with this package. The scipy algorithms are the
Nelder-Mead, Powell, SLSQP, COBYLA and BFGS algorithms.

Output: a result object returned from the algorithm
If the result is from a scipy algorithm, then it contains the following
attributes:
x: ndarray
    The solution of the optimization.
success: bool
     Whether or not the optimizer exited successfully.
status: int Termination status of the optimizer.
    Its value depends on the underlying solver. Refer to message for details.
message: str
    Description of the cause of the termination.
fun, jac, hess, hess_inv: ndarray
    Values of objective function,
    Jacobian, Hessian or its inverse (if available). The Hessians
    may be approximations, see the documentation of the function in question.
nfev, njev, nhev: int
    Number of evaluations of the objective
    functions and of its Jacobian and Hessian.
nit: int
    Number of iterations performed by the optimizer.
maxcv: float
    The maximum constraint violation.

For the attributes of the result object returned by the DFO method see
the README file in the DFO_src directory.

author: Anahita Hassanzadeh
email: Anahita.Hassanzadeh@gmail.com
"""
from scipy.optimize import minimize
from .DFO_src.dfo_tr import dfo_tr
import time

def bb_optimize(func, x_0, alg, options=None):
    # start timing
    start = time.time()

    # call the specified algorithm with the given points and options.
    if alg.lower() == 'dfo':
        res = dfo_tr(func, x_0, options=options)
    else:
        res = minimize(func, x_0, method=alg, options=options)

    end = time.time()

    # print out the time and return the result object
    print "Total time is {} seconds with " .format(end - start) + alg + (
        " method.")
    return res
