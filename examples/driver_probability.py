import numpy as np
from matplotlib import pyplot as plt
from goph420_lab01 import integration as Itg

# Gauss-Legendre quadrature method:

# 8. i

P = np.zeros(5)
eps_a1 = np.zeros(4)

Z = 4.0
A = Itg.GaussDist(mu=1.5, sigma=0.5)
lims = [1.5, Z]
npts = [1, 2, 3, 4, 5]

for i in range(0, 5):
    I = Itg.integrate_gauss(A, lims, npts[i])
    P[i] = (abs(0.5 - I)/0.5)

for j in range(0, 4):
    eps_a1[j] = np.abs((P[j+1] - P[j])/P[j+1])

plt.loglog([1, 2, 3, 4, 5], P)
plt.title("Absolute relative error convergence graph")
plt.xlabel("Number of points (n)")
plt.ylabel("Absolute relative error.")
plt.grid(min)
plt.show()

# 8. ii
P2 = np.zeros(5)
eps_a2 = np.zeros(4)
A = Itg.GaussDist(mu=10.28, sigma=0.05)
lims = [10.25, 10.35]
npts = [1, 2, 3, 4, 5]

for i in range(0, 5):
    P2[i] = Itg.integrate_gauss(A, lims, npts[i])

for j in range(0, 4):
    eps_a2[j] = np.abs((P2[j+1] - P2[j])/P2[j+1])

plt.loglog([2, 3, 4, 5], eps_a2)
plt.title("Absolute relative error convergence graph")
plt.xlabel("Number of points (n)")
plt.ylabel("Approximate relative error.")
plt.grid(min)
plt.show()
