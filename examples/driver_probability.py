import numpy as np
from matplotlib import pyplot as plt
from goph420_lab01 import integration as Itg

# Gauss-Legendre quadrature method:
dt = 0.04
x = np.arange(0.0, 10 + dt, dt)
mu = 0
sigma = 1
A = Itg.GaussDist(mu, sigma)
lims = [0, 1]
npts = 3

print(x)
fx = Itg.integrate_gauss(A, lims, npts)
