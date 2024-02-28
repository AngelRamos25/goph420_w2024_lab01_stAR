import numpy as np
from matplotlib import pyplot as plt
from goph420_lab01 import integration as Itg


def plot_raw_cut(t, v):
    N = len(v)
    v2 = np.abs(v)
    vmax = max(v2)

    for i in range(2, N):
        if v2[i] > 0.005*vmax:
            index = i

    T = t[index]
    fig, C = plt.subplots(1, 1)
    plt.plot(t, v)
    plt.axvline(x=T, color='b')

    plt.title("time vs S-wave velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("S-Wave velocity (mm/s)")
    plt.legend(['S-wave velocity', f" Data cut at {T} s"])
    plt.xlim([-0.1, 10])
    plt.grid(min)
    plt.savefig(
        'C:/Users/mange/Desktop/UoC/Winter 2024/GOPH_420/goph420_w2024_lab01_stAR/figures/raw_data_cut.pdf')
    plt.show()
    return index


def plot_convergence_error(t, v, index):
    T = t[index]
    t = t[0:index]
    v2 = (1/T)*v[0:index]**2
    Nv2 = len(v2)

    Ns = 9
    div = 200
    trap = np.zeros(Ns)
    simp1 = np.zeros(Ns)
    dtSaved = np.zeros(Ns)

    for x in range(0, Ns):

        div *= 0.5
        dt = 1/(div)
        dtSaved[x] = dt
        nD = 2**x
        tT = np.arange(0.0, T, dt)
        pos = np.arange(0, Nv2, nD)
        vT = v2[pos]
        if len(tT) != len(vT):
            tT = np.arange(0.0, T+dt, dt)

        trap[x] = Itg.integrate_newton(tT, vT, 'trap')
        simp1[x] = Itg.integrate_newton(tT, vT, 'simp')
        print([trap[x], simp1[x]])

    E_trap = np.zeros(Ns-1)
    E_simp1 = np.zeros(Ns-1)

    for i in range(1, Ns):
        E_trap[i-1] = abs((trap[i] - trap[i-1])/trap[i])
        E_simp1[i-1] = abs((simp1[i] - simp1[i-1])/simp1[i])

    fig, P = plt.subplots(1, 1)
    plt.loglog(dtSaved[1:Ns], E_trap)
    plt.loglog(dtSaved[1:Ns], E_simp1)
    plt.title("Approximate relative error convergence graph")
    plt.xlabel("dt-sampling (s)")
    plt.ylabel("Approximate relative error.")
    plt.legend(['Trapezoid', "Simpson's"])
    plt.grid(min)
    plt.savefig(
        'C:/Users/mange/Desktop/UoC/Winter 2024/GOPH_420/goph420_w2024_lab01_stAR/figures/convergence_error.pdf')
    plt.show()
    return ()


if __name__ == "__main__":
    t, v = np.loadtxt('s_wave_data.txt', float, unpack=True)
    index = plot_raw_cut(t, v)
    plot_convergence_error(t, v, index)
