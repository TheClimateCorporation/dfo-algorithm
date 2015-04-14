# dfo-algorithm
This package provides an implementation of the derivative-free
optimization algorithm, DFO, developed by A. Conn, K. Scheinberg,
L. Vicente.
Using this package, the user can solve a derivative-free 
blackbox optimization problem with the DFO method as well 
as five derivative free algorithms from the scipy.optimize library. 
The scipy algorithms are the Nelder-Mead, Powell, SLSQP,
COBYLA and BFGS algorithms.

To run a set of sample problems, the user can call the
“run_test_func.py” module.

To solve a user-defined problem:
==========================================================
  - Write your blackbox optimization function in a new Python
  module. For examples of such functions see
  blackbox_opt/test_funcs/funcs_def.py
  - Write a run file with the module you wrote in the previous 
  step imported. You can modify the run_test_func.py module and
  use it as your run file. Note that you should specify the
  algorithms(s) that you wish to solve your problems with,
  a starting point and suitable options.
