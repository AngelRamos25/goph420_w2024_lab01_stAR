import numpy as np
from matplotlib import pyplot as plt
from goph420_lab01 import integration as Itg

# Gauss-Legendre quadrature method:

# 8. i

# Calculates probability for event 1) using Gauss quadrature methos:


def probability_event_1():

    P = np.zeros(5)
    eps_a1 = np.zeros(4)

    Z = 4.0
    A = Itg.GaussDist(mu=1.5, sigma=0.5)
    lims = [1.5, Z]
    npts = [1, 2, 3, 4, 5]

    for i in range(0, 5):
        I = Itg.integrate_gauss(A, lims, npts[i])
        P[i] = abs(0.5 - I)

    for j in range(0, 4):
        eps_a1[j] = np.abs((P[j+1] - P[j])/P[j+1])

    fig, C = plt.subplots(1, 1)
    plt.loglog([1, 2, 3, 4, 5], P)
    plt.title("Convergence error - Gauss-Legendre queadrature.")
    plt.xlabel("Number of points (n)")
    plt.ylabel("Absolute relative error.")
    plt.grid(min)

    # Saving image. This section is commented since it could be an error if run in other computer.
    # plt.savefig(
    #    'C:/Users/mange/Desktop/UoC/Winter 2024/GOPH_420/goph420_w2024_lab01_stAR/figures/Q1_Gauss.pdf')
    plt.show()
    return (P)

# 8. ii

# Calculates probability for event 2) using Gauss quadrature methos:


def probability_event_2():
    P2 = np.zeros(5)
    eps_a2 = np.zeros(4)
    A = Itg.GaussDist(mu=10.28, sigma=0.05)
    lims = [10.25, 10.35]
    npts = [1, 2, 3, 4, 5]

    for i in range(0, 5):
        P2[i] = Itg.integrate_gauss(A, lims, npts[i])

    for j in range(0, 4):
        eps_a2[j] = np.abs((P2[j+1] - P2[j])/P2[j+1])

    fig, C2 = plt.subplots(1, 1)
    plt.loglog([2, 3, 4, 5], eps_a2)
    plt.title("Convergence error - Gauss-Legendre quadrature.")
    plt.xlabel("Number of points (n)")
    plt.ylabel("Approximate relative error.")
    plt.grid(min)

    # Saving image. This section is commented since it could be an error if run in other computer.
    plt.savefig(
        'C:/Users/mange/Desktop/UoC/Winter 2024/GOPH_420/goph420_w2024_lab01_stAR/figures/Q2_Gauss.pdf')
    plt.show()
    return (P2)


if __name__ == "__main__":
    P1 = probability_event_1()
    P2 = probability_event_2()
    print(f"Probabilitys of event 1: {P1}")
    print(f"Probabilitys of event 2: {P2}")
