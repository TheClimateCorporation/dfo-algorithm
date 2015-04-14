"""
Description: This file contains functions to optimize the trust-region
subproblem that is solved sequentially within the DFO method. 
author: Anahita Hassanzadeh
email: Anahita.Hassanzadeh@gmail.com
"""
from scipy import linalg as LA
import numpy as np
from scipy.optimize import fmin_cobyla

def trust_sub(g, H, delta):
    """
    This function solves the trust region subproblem when the
    Frobenuis norm of H is not very small.
    The subproblem is:
        min g.T s + 1/2 s.T H s
        s.t. || s || <= delta

    Note that any restriction that the problem has
    can be added to the constriants in the trust region.
    In that case the following algorithm will not work and
    another package should be used. The alternative is to
    penalize the constraints violations in the objective function
    evaluations.
    """
    def obj_func(s):
        'objective function of the trust region subproblem'
        return (np.dot(g.T, s) + reduce(np.dot, [(.5*s).T, H, s]))  
    
    def radius_cons(s):
        'the trust region consraint ||s|| <= delta'
        nrms = LA.norm(s)
        return (delta - nrms) 
    
    n = len(g)    
    s0 = np.zeros((n, 1))   
    s = fmin_cobyla(func=obj_func, x0=s0, cons=radius_cons, 
                   iprint=0, rhoend=1e-9, maxfun=10e4)
    return (s.reshape(n, 1), obj_func(s).reshape(1, 1))