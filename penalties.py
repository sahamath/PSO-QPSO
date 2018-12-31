import numpy as np

def l1_equality_penalty(diff, tolerance):
    k = 10e12 # penalty factor: tune this penalty factor depending on the range of the function you are trying to maximize
    abs_diff = np.abs(diff)
    if abs_diff <= tolerance:
        # this means 1 - (alpha + beta) is in interval [-tolerance, +tolerance]
        # since the solution is within constraint, no penalty is incurred
        return 0
    else:
        # solution violates constraint and hence penalty is incurred
        return k*(abs_diff)

def l1_inequality_penalty(x, error):
    k = 10e12 # penalty factor
    if x + error <= 0:
        return 0
    else:
        abs_x = np.abs(x)
        return k*(abs_x)

def l2_equality_penalty(diff, tolerance):
    k = 10e100# penalty factor: tune this penalty factor depending on the range of the function you are trying to maximize
    abs_diff = np.abs(diff)
    if abs_diff <= tolerance:
        # this means 1 - (alpha + beta) is in interval [-tolerance, +tolerance]
        # since the solution is within constraint, no penalty is incurred
        return 0
    else:
        # solution violates constraint and hence penalty is incurred
        return 3*k*(abs_diff)

def l2_inequality_penalty(x, error):
    k = 10e100 # penalty factor
    if x + error <= 0:
        return 0
    else:
        abs_x = np.abs(x)
        return k*(abs_x**2)

def modified_inequality_penalty(x):
    k = 10e100
    e = 10e-12
    if x - e < 0:
        return 0
    else:
        abs_x = np.abs(x)
        return k*(abs_x**2)
