import numpy as np
""" Integration functions 
    This file contains certain functions to calculate an integral applying the
    Newton-Cotes methods and Gauss-Legendre quadrature method
"""


def integrate_newton(
        x: float,
        f: float,
        alg: str = 'trap') -> any:
    """ Newton-cotes methods:
        This function performs a numerical integration using either trapezoidal rule (trap),
        1/3 Simpson's rule (simp1/3), or 3/8 Simpson's rule (simp3/8)

        Inputs:
        ---------------
        x: x points where f(x) is evaluated or measured.
        f: value f(x)
        alg: type of algorithm desires. 'trap' = trapezoidal rule, 'simp1/3' = 1/3 Simpson's rule, and 'simp3/8' Simpson's 3/8 rule.

        Outputs:
        ---------------
        A: Returns the area of the curve f(x)
    """

    dx = x[1] - x[0]
    Nf = len(f)

    if alg.lower() == 'trap':

        A = (dx/2)*(f[0] + 2*sum(f[1:Nf-2]) + f[Nf-1])

    elif alg.lower() == 'simp1/3':

        evenN = range(2, Nf-1, 2)
        oddN = range(1, Nf-1, 2)
        A = (dx/3)*(f[0] + 4*sum(f[oddN]) + 2*sum(f[evenN]) + f[Nf-1])

    elif alg.lower() == 'simp3/8':

        threeN = np.arange(3, Nf-1, 3)
        restN = np.arange(1, Nf-1, 1)
        restN = np.delete(restN, threeN-1)
        A = (3*dx/8)*(f[0] + 3*sum(f[restN]) + 2*sum(f[threeN]) + f[Nf-1])

    return A
