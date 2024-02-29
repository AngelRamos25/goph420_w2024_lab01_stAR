import numpy as np


def integrate_newton(
        x: float,
        f: float,
        alg: str = 'trap') -> any:
    """ Newton-cotes methods:
    This function performs a numerical integration using either trapezoidal rule (trap),
    1/3 Simpson's rule (simp1/3), and 3/8 Simpson's rule (simp3/8)

    Inputs:
    ---------------
    x: x points where f(x) is evaluated or measured.
    f: value f(x)
    alg: type of algorithm desires. 'trap' = trapezoidal rule, 'simp' = Simpson's rule and Simpson's 3/8 rule.

    Outputs:
    ---------------
    A: Returns the area of the curve f(x)

    Raises:
    -----------------
    ValueError if alg contains a str other than these three options [trap, sim1/3, simp3/8].
    ValueError if the dimensions of x and f are incompatible. 
    """

    alg = alg.lower().strip()
    options = ['trap', 'simp']
    if alg not in options:
        raise ValueError(
            "Not valid method, please select an option from the following list: trap, simp1/3, or simp3/8."
        )

    Nf = len(f)
    Nx = len(x)
    if Nf != Nx:
        raise ValueError(
            "Different lenghts on vectors x and f(x), please make sure these vectors have same lenght."
        )
    dx = x[1] - x[0]

    if alg == 'trap':
        A = (dx/2)*(f[0] + 2*sum(f[1:Nf-2]) + f[Nf-1])

    if alg == 'simp':
        if Nf == 2:
            raise ValueError(
                "Simpsons rule requieres at least 3 points."
            )

        if (Nf % 2 == 0 and Nf == 4):

            A = (3*dx/8)*(f[0] + 3*f[1] + 3*f[2] + f[3])

        elif (Nf % 2 == 0 and Nf > 4):

            A = (3*dx/8)*(f[0] + 3*f[1] + 3*f[2] + f[3])
            fm = f[3:-1]
            evenN = range(2, len(fm)-1, 2)
            oddN = range(1, len(fm)-1, 2)
            A = A + (dx/3)*(fm[0] + 4*sum(fm[oddN]) +
                            2*sum(fm[evenN]) + fm[-1])

        if (Nf % 2 == 1 and Nf == 3):

            A = (dx/3)*(f[0] + 4*f[1] + f[2])

        elif (Nf % 2 == 1 and Nf > 3):

            evenN = range(2, Nf-1, 2)
            oddN = range(1, Nf-1, 2)
            A = (dx/3)*(f[0] + 4*sum(f[oddN]) + 2*sum(f[evenN]) + f[-1])

    return A


class GaussDist:
    def __init__(self, mu: float = 0, sigma: float = 1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        z = np.subtract(x, self.mu)/self.sigma
        f = (1/(np.sqrt(2*np.pi*self.sigma**2))) * np.exp((-z**2) / 2)
        return f


def integrate_gauss(
        f,
        lims: float,
        npts: int = 3):
    """ Gauss-Legendre quadrature method:
    This function performs a numerical integration using Gauss-Legendre quadrature method.

    Inputs:
    ---------------
    f: calleable function - Gaussian distribution - default = mean = 0, std = 1.
    lims: Integral limits [a, b]
    npts: Number of points to use in the Gauss quadrature method.

    Outputs:
    ---------------
    I: Returns the probability (integral) of an event z.

    Raises:
    -----------------
    TypeError if f is not callable.
    ValueError if lims does not have len == 2.
    ValueError if lims[0] or lims[1] are not convertible to float.
    ValueError if npts is not in [1, 2, 3, 4, 5].
    """

    if callable(f) is False:
        raise TypeError(
            "f is not calleable"
        )

    options = [1, 2, 3, 4, 5]
    if npts not in options:
        raise ValueError(
            "npts is not in [1, 2, 3, 4, 5]"
        )
    if len(lims) != 2:
        raise ValueError(
            "Not enough or too many values on lims."

        )
    float(lims[0])
    float(lims[1])

    if npts == 1:
        ck = 2
        sk = 0
    if npts == 2:
        ck = [1, 1]
        sk = [-1/np.sqrt(3), 1/np.sqrt(3)]

    elif npts == 3:
        ck = [5/9, 8/9, 5/9]
        sk = [-np.sqrt(3/5), 0, np.sqrt(3/5)]
    elif npts == 4:
        ck = [(18-np.sqrt(30))/36, (18+np.sqrt(30))/36,
              (18+np.sqrt(30))/36, (18-np.sqrt(30))/36]
        sk = [-(np.sqrt(3/7 + (2/7)*np.sqrt(6/5))), -(np.sqrt(3/7 - (2/7)*np.sqrt(6/5))),
              (np.sqrt(3/7 - (2/7)*np.sqrt(6/5))), (np.sqrt(3/7 + (2/7)*np.sqrt(6/5)))]
    elif npts == 5:
        ck = [(322 - 13*np.sqrt(70))/900, (322 + 13*np.sqrt(70))/900,
              128/225, (322 + 13*np.sqrt(70))/900,  (322 - 13*np.sqrt(70))/900]
        sk = [-(1/3)*(np.sqrt(5 + 2*np.sqrt(10/7))), -(1/3)*(np.sqrt(5 - 2*np.sqrt(10/7))),
              0, (1/3)*(np.sqrt(5 - 2*np.sqrt(10/7))), (1/3)*(np.sqrt(5 + 2*np.sqrt(10/7)))]

    wk = np.multiply((lims[1] - lims[0])/2, ck)
    xk = (lims[0] + lims[1])/2 + np.multiply((lims[1] - lims[0])/2, sk)
    fk = f(xk)
    I = np.sum(np.multiply(wk, fk))

    return I


class Polyn:
    def __init__(self, npts):
        self.npts = npts

    def __call__(self, x):
        if self.npts == 1:
            fx = x + 1
        if self.npts == 2:
            fx = x**3 + x**2 + x + 1
        if self.npts == 3:
            fx = x**5 + x**4 + x**3 + x**2 + x + 1
        if self.npts == 4:
            fx = x**7 + x**6 + x**5 + x**4 + x**3 + x**2 + x + 1
        if self.npts == 5:
            fx = x**9 + x**8 + x**7 + x**6 + x**5 + x**4 + x**3 + x**2 + x + 1
        return fx
