"""
Description: An implementation of the DFO algorithm developed by A. Conn,
K. Scheinberg, L. Vicente.
author: Anahita Hassanzadeh
email: Anahita.Hassanzadeh@gmail.com
"""
import numpy as np
from scipy import linalg as LA
import time
from quad_Frob import quad_Frob
from trust_sub import trust_sub

class params:
    """
    The class of parameters for the trust region algorithms
    The parameters include the initial delta, its expansion and contraction
    factors and acceptance as well as the maximum number of iterations
    """
    def __init__(self):
        self.init_delta = 1.0  # initial delta (i.e. trust region radius)
        self.tol_delta = 1e-10  # smallest delta to stop
        self.max_delta = 100.0  # max possible delta
        # TR radius adjustment parameters
        self.gamma1 = 0.8  # radius shrink factor
        self.gamma2 = 1.5  # radius expansion factor

        self.eta0 = 0.0   # step acceptance test (pred/ared) threshold
        self.eta1 = 0.25  # (pred/ared) level to shrink radius
        self.eta2 = 0.75  # (pred/ared) level to expand radius

        self.tol_f = 1e-15  # threshold for abs(fprev - fnew)- used to stop
        self.tol_norm_g = 1e-15  # threshold for norm of gradient- used to stop
        self.tol_norm_H = 1e-10  # threshold for the (frobenius) norm of H
        self.maxfev = 1000  # maximum number of iterations

        self.min_del_crit = 1e-8  # minimum delta for the criticality step
        self.min_s_crit = 0.1  # minimum step for the criticality step

class result:
    def __init__(self, x, f, iteration, iter_suc, func_eval, delta):
        self.x = x
        self.fun = f
        self.iteration = iteration
        self.iter_suc = iter_suc
        self.func_eval = func_eval
        self.delta = delta

def _set_parameters(options):
    par = params()
    if options:
        for key in options:
            if (key not in par.__dict__.keys() and
                key not in ["sample_gen", "verbosity"]):
                raise ValueError("{!r} is not a valid option".format(key))
            par.__dict__[key] = options[key]
    return par

def _build_initial_sample(x_initial, delta, options=None):
    """
    Given the original point and delta, generate a sample set to start the
    method. If nothing is provided here, the algorithm works fine.
    Note that if the number of points are greater than (n+1)(n+2)/2, then
    a point is only be substituted and no additional points will be added.
    It may be a good idea to leave some point generation to the model so it
    uses the min Frobenius norm model in the quad_Frob module.
    Do not build a sample size more than (n+1)(n+2)/2. We are going to build a
    quadratic model after all!
    """
    n = x_initial.shape[0]
    # if the user hasn't specified any options or no option for sample build
    # set it to auto.
    if not options or "sample_gen" not in options.keys():
        option = "auto"
    else:
        option = options["sample_gen"]

    if option == "auto":
        Y = np.tile(x_initial, (1, 2*n)) + (0.5 * delta * np.hstack(
            (np.eye(n), -np.eye(n))))
        Y = np.hstack((x_initial, Y))
    elif option == "manual":
        # othewise the user can specify the starting points manually
        # for example, for a function with 4 variables we have:
        Y1 = np.array([150.00, 150.00, 230.00, 150.00]).reshape((n, 1))
        Y2 = np.array([150.00, 8.00, 230.00, 250.00]).reshape((n, 1))
        Y3 = np.array([150.00, 50.00, 230.00, 230.00]).reshape((n, 1))
        Y = np.hstack((x_initial, Y1, Y2, Y3))
    else:
        raise ValueError("The sample option should be manual or auto.")

    nY = Y.shape[1]
    return (Y, nY)

def _shift_sort_points(x, Y, nY, f_values):
    '''shift the points to the origin and sort
    the corresponding objective values and norms
'''
    Ynorms = Y - np.tile(x, (1, nY))
    Ynorms = np.sqrt(np.sum(Ynorms**2, axis=0))
    index = np.argsort(Ynorms)
    Ynorms = Ynorms[index]
    Y = Y[:, index]
    f_values = f_values[index]
    return (Ynorms, Y, f_values)

def dfo_tr(bb_func, x_initial, options=None):
    """
    This is the main function of the trust region method. It takes the
    intialized point as the input and returns the solution x and its
    corresponding value. If the maximum number of allowed iterations is
    set to a large enough number, the method returns a stationary point of
    the optimziation problem. The main functions inside this function are
    quad_Frob and trust functions. For the details of the algorithm see
    chapter 11 of the Introduction to DFO by Conn, Scheinberg and Vicente

    arguments
    ---------
    bb_func: function object
        The blackbox function to minimize. The function takes a numpy.ndarray
        and returns a scalar. For examples of this see test_funcs/funcs_def.py
    x_initial: np.ndarray
        a starting point for the algorithm
    options: dict
        See the README file for the supported options.

    returns
    ---------
    a "result object" with the following attributes
        x: np.ndarray
            the optimal point found
        fun: float
            the optimal value
        iteration: int
            number of iterations
        iter_suc: int
            number of successful iterations (where the objective decreased)
        func_eval: int
            number of blackbox function evaluations
        delta: float
            the radius of the trust region at termination
    """
    # start timing and set the paramters
    start_time = time.time()

    par = _set_parameters(options)
    # see the param class for the description of parameters
    delta = par.init_delta
    tol_delta = par.tol_delta
    max_delta = par.max_delta
    gamma1 = par.gamma1
    max_iter = par.maxfev
    eta0 = par.eta0
    eta1 = par.eta1  # this may be used depending on the application
    eta2 = par.eta2
    tol_f = par.tol_f
    gamma2 = par.gamma2
    tol_norm_g = par.tol_norm_g
    tol_norm_H = par.tol_norm_H
    min_del_crit = par.min_del_crit
    min_s_crit = par.min_s_crit

    # set the verbosity parameter
    if options and "verbosity" in options:
        verb = options["verbosity"]
        if verb == 0 or verb == 1:
            verbosity = verb
        else:
            raise ValueError("verbosity option should be 0 or 1")
    else:
        verbosity = 1
    # find the dimension of the problem
    # find the max and min number of points for the quadratic model
    n = x_initial.shape[0]
    maxY = (n+1) * (n+2) / 2
    minY = n+1

    # iterations counters
    iteration = iter_suc = func_eval = 0

    # construct the intial sample set and evaluate the objective at them
    # by evaluating the blackbox function
    x_initial = x_initial.reshape((n, 1))
    Y, nY = _build_initial_sample(x_initial, delta, options)
    f_values = np.empty((nY, 1))

    for i in range(nY):
        f_values[i] = bb_func(Y[:, i].reshape(n, 1))
        func_eval += 1

    # find the point with minimum objective. set it as the first center
    ind_f_sort = np.argsort(f_values[:, 0])[0]
    x = Y[:, ind_f_sort].reshape(n, 1)
    f = f_values[ind_f_sort][0]

    # print out the initial evaluation results
    if verbosity:
        print ("\n Iteration Report \n")
        print ('|it |suc|  objective  | TR_radius  |    rho    | |Y|  \n')
        print ("| {} |---| {} | {} | --------- | {} \n".format(iteration,
               format(f, '0.6f'), format(delta, '0.6f'), nY))

    # Start the TR main loop
    while True:
        success = 0

        # construct the quadratic model
        H, g = quad_Frob(Y, f_values)

        # obtain the norm of the "g" (coeff of linear components) of the model
        normg = LA.norm(g)

        # Stopping criterion --stop if the TR radius or norm of g is too small
        if normg <= tol_norm_g or delta < tol_delta:
            break

        # Start the TR iteration
        # Minimize the model on the TR, aka: solve the TR subproblem
        if LA.norm(H, 'fro') > tol_norm_H:  # make sure the trust can work
            s, val = trust_sub(g, H, delta)
        else:   # otherwise take a steepest descent step less than delta
            s = -(delta/normg) * g
            val = np.dot(g.T, s) + 0.5 * reduce(np.dot, [s.T, H, s])

        # f: model value at the origin of trust region
        # Evaluate the model value at the new point.
        fmod = f + val

        # Stop if the reduction is very small or max iterations reached
        if abs(val) < tol_f or iteration > max_iter:
            break

        # Take the step
        xtrial = x + s

        # Evaluate the function at the new point.
        ftrue = bb_func(xtrial)
        func_eval += 1

        #  Calculate ared/pred.
        pred = f - fmod
        ared = f - ftrue
        rho = ared / pred

        # Updating iterate and trust-region radius.
        if rho >= eta0 and f - ftrue > 0:
            # Set new iterate if there is success.
            success = 1
            iter_suc += 1

            # Update the center of the trust region
            x = xtrial
            f = ftrue
            if rho >= eta2:  # If confident, increase delta
                delta = min(gamma2*delta, max_delta)
        else:
            if nY >= minY:
                # decrease the TR radius
                delta = gamma1*delta

        iteration += 1

        # print iteration report
        if verbosity:
            print ("| {} | {} | {:.6f} | {} | {} | {} \n".format(iteration,
                        success, float(f), format(delta, '0.6f'),
                        format(rho[0][0], '0.6f'), nY))
            print x.T, "\n"

        # order the sample set according to the distance to the
        # current iterate.
        Ynorms, Y, f_values = _shift_sort_points(x, Y, nY, f_values)

        # Update the sample set.
        if success:
            reached_max = 1
            if nY < maxY:
                reached_max = 0

            # add or substitute the furthest point
            if reached_max:  # substitute
                Y[:, nY-1] = xtrial.T
                f_values[nY-1] = ftrue

            else:  # add
                nY = nY + 1
                Y = np.hstack((Y, xtrial))
                f_values = np.vstack((f_values, np.array([ftrue])))
        else:  # if not successful
            if nY >= maxY:
                # add if closer than the furthest point, otherwise
                # keep the set as it is
                if LA.norm(xtrial-x) <= Ynorms[nY-1]:
                    # Substitute the furthest point
                    Y[:, nY-1] = xtrial.T
                    f_values[nY-1] = ftrue
            else:
                nY = nY + 1
                Y = np.hstack((Y, xtrial))
                f_values = np.vstack((f_values, np.array([ftrue])))

        # shift and sort the set of points once more
        Ynorms, Y, f_values = _shift_sort_points(x, Y, nY, f_values)

        # implementation of an idea similar to the criticality step
        # this is, if the gradient is small, then delta should be
        # proportional to it -- should not happen often in practice
        if delta < min_del_crit and LA.norm(s) < min_s_crit:
            # only keep the points that are within 100*delta radius
            selection = np.where(Ynorms < delta * 100)
            Y = Y[:, selection[0]]
            f_values = f_values[selection]
            nY = Y.shape[1]

    end_time = time.time()
    if verbosity:
        # Final Report
        print ('*****************REPORT************************\n')
        print ("Total time is {} seconds.\n".format(end_time - start_time))
        print ("Norm of the gradient of the model is {}.\n".format(normg))
        print ('***************Final Report**************\n')
        print ("|iter | #success| #fevals| final fvalue | final tr_radius|\n")
        print ("| {} |    {}   |   {}   |   {}   |  {}  \n"
                .format(iteration, iter_suc, func_eval, f, delta))
    res = result(x, f, iteration, iter_suc, func_eval, delta)
    return res