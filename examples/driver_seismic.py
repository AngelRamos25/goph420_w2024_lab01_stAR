import numpy as np
from matplotlib import pyplot as plt

dt = 0.01  # Time sampling in seconds.
t, v = np.loadtxt('s_wave_data.txt', float, unpack=True)

N = len(v)
v2 = np.abs(v)
tmax = max(v2)


for i in range(2, N):
    if (v2[i] > 0.005*tmax):
        index = i


T = t[index]
print(T)

plt.plot(T, 0, marker='o')
plt.plot(t, v2)

plt.title("time vs S-wave velocity")
plt.xlabel("Time (s)")
plt.ylabel("S-Wave velocity (mm/s)")
plt.legend(['V1'])
plt.xlim([-0.1, 10])
plt.grid(min)
plt.show()

t = t[0:index]
v = (1/T)*v[0:index]**2

N = len(v)

# Trapezoid rule:

I = integrate_newton(t, v, 'trap')
print(I)
I1 = integrate_newton(t, v, 'simp3/8')
print(I1)

# 1/3 Simpson rule
