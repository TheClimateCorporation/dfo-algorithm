"""
Description: This file contains functions to optimize the trust-region
subproblem that is solved sequentially within the DFO method.
Author: Anahita Hassanzadeh
Email: Anahita.Hassanzadeh@gmail.com
"""
from scipy import linalg as LA
import numpy as np
from scipy.sparse import csr_matrix

def _secular_eqn(lambda_0, eigval, alpha, delta):
    """
    The function secular_eqn returns the value of the secular
    equation at a set of m points.
    """
    m = lambda_0.size
    n = len(eigval)
    unn = np.ones((n, 1))
    unm = np.ones((m, 1))
    M = np.dot(eigval, unm.T) + np.dot(unn, lambda_0.T)
    MC = M.copy()
    MM = np.dot(alpha, unm.T)
    M[M != 0.0] = MM[M != 0.0] / M[M != 0.0]
    M[MC == 0.0] = np.inf * np.ones(MC[MC == 0.0].size)
    M = M*M
    value = np.sqrt(unm / np.dot(M.T, unn))

    if len(value[np.where(value == np.inf)]):
        inf_arg = np.where(value == np.inf)
        value[inf_arg] = np.zeros((len(value[inf_arg]), 1))

    value = (1.0/delta) * unm - value

    return value

def rfzero(x, itbnd, eigval, alpha, delta, tol):
    """
    This function finds the zero of a function
    of one variable to the RIGHT of the starting point x.
    The code contanins a small modification of the M-file fzero in matlab,
    to ensure a zero to the right of x is searched for.
    """
    # start the iteration counter
    itfun = 0

    # find the first three points, a, b, and c and their values
    if (x != 0.0):
        dx = abs(x) / 2
    else:
        dx = 0.5

    a = x
    c = a
    fa = _secular_eqn(a, eigval, alpha, delta)
    itfun = itfun + 1

    b = x + dx
    b = x + 1
    fb = _secular_eqn(b, eigval, alpha, delta)
    itfun = itfun + 1

    # find change of sign
    while ((fa > 0) == (fb > 0)):

        dx = 2*dx

        if ((fa > 0) != (fb > 0)):
            break
        b = x + dx
        fb = _secular_eqn(b, eigval, alpha, delta)
        itfun = itfun + 1

        if (itfun > itbnd):
            break

    fc = fb

    # main loop, exit from the middle of the loop
    while (fb != 0):
        # Ensure that b is the best result so far, a is the previous
        # value of b, and c is on hte opposit side of 0 from b
        if (fb > 0) == (fc > 0):
            c = a
            fc = fa
            d = b - a
            e = d

        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        # convergence test and possible exit
        if itfun > itbnd:
            break

        m = 0.5 * (c-b)
        rel_tol = 2.0 * tol * max(abs(b), 1.0)

        if (abs(m) <= rel_tol) or (abs(fb) < tol):
            break

        # choose bisection or interpolation
        if (abs(e) < rel_tol) or (abs(fa) <= abs(fb)):
            # bisection
            d = e = m
        else:
            # interpolation
            s = float(fb)/fa
            if a == c:
                # linear interpolation
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                # Inverse quad interpolation
                q = float(fa)/fc
                r = float(fb)/fc
                p = s * (2.0 * m * q * (q-r) - (b-a) * (r-1.0))
                q = (q-1.0) * (r-1.0) * (s-1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            # Check if the interpolated point is acceptable
            if (2.0*p < 3.0*m*q - abs(rel_tol*q)) and (p < abs(0.5*e*q)):
                e = d
                d = float(p)/q
            else:
                d = m
                e = m
            #  end of iterpolation

        # Next point
        a = b
        fa = fb
        if (abs(d) > rel_tol):
            b = b + d
        else:
            if b > c:
                b = b - rel_tol
            else:
                b = b + rel_tol

        fb = _secular_eqn(b, eigval, alpha, delta)
        itfun = itfun + 1

    return (b, c, itfun)


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

    tol = 10e-12
    tol_seqeq = 10e-8
    key = 0
    itbnd = 50
    lambda_0 = 0
    s_factor = 0.8
    b_factor = 1.2
    n = len(g)
    coeff = np.zeros((n, 1))

    # convert H to full matrix if sparse
    T = csr_matrix(H)
    T = T.todense()
    H = np.squeeze(np.asarray(T))

    # get the eigen value and vector
    D, V = LA.eigh(0.5 * (H.T + H))
    count = 0
    eigval = D[np.newaxis].T
    # find the minimum eigen value
    jmin = np.argmin(eigval)
    mineig = np.amin(eigval)

    # depending on H, find a step size
    alpha = np.dot(-V.T, g)
    sig = (np.sign(alpha[jmin]) + (alpha[jmin] == 0).sum())[0]

    # PSD case
    if mineig > 0:
        lambda_0 = 0
        coeff = alpha * (1/eigval)
        s = np.dot(V, coeff)
        # That is, s = -v (-v.T g./eigval)
        nrms = LA.norm(s)
        if nrms < b_factor*delta:
            key = 1
        else:
            laminit = np.array([[0]])
    else:
        laminit = -mineig

    # Indefinite case
    if key == 0:
        if _secular_eqn(laminit, eigval, alpha, delta) > 0:
          b, c, count = rfzero(laminit, itbnd, eigval, alpha, delta, tol)

          if abs(_secular_eqn(b, eigval, alpha, delta)) <= tol_seqeq:
              lambda_0 = b
              key = 2
              lam = lambda_0 * np.ones((n, 1))

              coeff, s, nrms, w = compute_step(alpha, eigval, coeff, V, lam)

              if (nrms > b_factor * delta or nrms < s_factor * delta):
                  key = 5
                  lambda_0 = -mineig
          else:
                key = 3
                lambda_0 = -mineig
        else:
            key = 4
            lambda_0 = -mineig

        lam = lambda_0 * np.ones((n, 1))

        if key > 2:
            arg = abs(eigval + lam) < 10 * (np.finfo(float).eps *
                np.maximum(abs(eigval), np.ones((n,1))))
            alpha[arg] = 0.0

        coeff, s, nrms, w = compute_step(alpha, eigval, coeff, V, lam)

        if key > 2 and nrms < s_factor * delta:
            beta = np.sqrt(delta**2 - nrms**2)
            s = s + reduce(np.dot, [beta, sig, V[:, jmin]]).reshape(n, 1)

        if key > 2 and nrms > b_factor * delta:
            b, c, count = rfzero(laminit, itbnd, eigval, alpha, delta, tol)
            lambda_0 = b
            lam = lambda_0 * np.ones((n, 1))

            coeff, s, nrms, w = compute_step(alpha, eigval, coeff, V, lam)

    # return the model prediction of the change in the objective with s
    val = np.dot(g.T, s) + reduce(np.dot, [(.5*s).T, H, s])

    return (s, val)

def compute_step(alpha, eigval, coeff, V, lam):
    w = eigval + lam
    arg1 = np.logical_and(w == 0, alpha == 0)
    arg2 = np.logical_and(w == 0, alpha != 0)
    coeff[w != 0] = alpha[w != 0] / w[w != 0]
    coeff[arg1] = 0
    coeff[arg2] = np.inf
    coeff[np.isnan(coeff)] = 0
    s = np.dot(V, coeff)
    nrms = LA.norm(s)
    return(coeff, s, nrms, w)
