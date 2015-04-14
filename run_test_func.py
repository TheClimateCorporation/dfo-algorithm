"""
Description:
This module runs six popular test problems for global optimization
that are defined and described in blackbox_opt/test_funcs/funcs_def.py.

The user is required to import a blackbox function, specify the
algorithm, a starting point and options.
            func: an imported blackbox function object
            x_0: starting point: numpy array with shape (n,1) --n is the
            dimension of x_0
            alg: selected algorithm: string
            options : a dictionary of options customized for each algorithm
For a full list of options available for DFO see the README file in
DFO_src directory.
For scipy algorithms, options for each alg are available by a call to
scipy.optimize.show_options. These options are available at
http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.show_options.html
author: Anahita Hassanzadeh
email: Anahita.Hassanzadeh@gmail.com
"""
import numpy as np
from blackbox_opt.bb_optimize import bb_optimize
# import the function to be optimized
from blackbox_opt.test_funcs.funcs_def import (ackley, sphere, rosen,
                                               booth, bukin, beale)
def get_results(func, x_0, alg, options):
    """
    This function calls the main blackbox optimization
    code with the specified algorithm, options and starting point and
    prints the best point found along with its optimal value.
    input: func: an imported blackbox function object
            x_0: starting point: numpy array with shape (n,1)
            alg: selected algorithm: string
            options : a dictionary of options customized for each algorithm
    """
    res = bb_optimize(func, x_0, alg, options)

    print "Printing result for function " + func.__name__ + ":"
    print "best point: {}, with obj: {:.6f}".format(
        np.around(res.x.T, decimals=5), float(res.fun))
    print "-------------" + alg + " Finished ----------------------\n"


if __name__ == "__main__":
    # separate the functions based on their dimension. This is merely
    # done to ensure the starting point x_0 will later have the
    # correct dimension
    nd_func_names = [sphere, rosen]  # functions from R^n -> R
    td_func_names = [ackley, booth, bukin, beale]   # functions from R^2 -> R
    all_func_names = td_func_names + nd_func_names

    # Run all the algorithms and problems with given starting points
    # Specify the starting point and options. For example, try the following
    # options.
    for func in all_func_names:
        if func in td_func_names:
            x_0 = np.array([1.3, 0.7])
        else:
            x_0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

        print "\n\n********* Function " + func.__name__ + "********"
        alg = "DFO"
        options = {"maxfev": 100, "init_delta": 20,
                   "tol_delta": 1e-25, "tol_f": 1e-26, "tol_norm_g": 1e-5,
                   "sample_gen": "auto", "verbosity": 0}
        get_results(func, x_0, alg, options)

        alg = "Powell"
        options = {"disp": True, "maxfev": 100, "ftol": 1e-26}
        get_results(func, x_0, alg, options)

        alg = "Nelder-Mead"
        options = {"disp": True, "maxfev": 100, "ftol": 1e-26}
        get_results(func, x_0, alg, options)

        alg = 'COBYLA'
        options = {"disp": True, "tol": 1e-25}
        get_results(func, x_0, alg, options)

        alg = 'BFGS'
        options = {"maxiter": 8, "disp": True, "gtol": 1e-5}
        get_results(func, x_0, alg, options)

        alg = 'SLSQP'
        options = {"maxiter": 20, "disp": True, "ftol": 1e-26}
        get_results(func, x_0, alg, options)
