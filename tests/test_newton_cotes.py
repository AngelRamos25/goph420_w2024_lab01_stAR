import unittest
import numpy as np
from goph420_lab01 import integration as Itg


class TestNewton(unittest.TestCase):

    def test_trap_integral(self):
        x = np.arange(0, np.pi/2, np.pi/1000)
        fx = np.cos(x)*np.sin(x)
        result = Itg.integrate_newton(x, fx, 'trap')
        self.assertAlmostEqual(result, 0.5, places=4)

        x = [0, 1, 2, 3]
        fx = [0, 1, 2, 3]
        result = Itg.integrate_newton(x, fx, 'trap')
        self.assertAlmostEqual(result, 4.5, places=4)

    def test_simp13_integral(self):
        x = np.arange(0, np.pi/2, np.pi/1000)
        fx = np.cos(x)*np.sin(x)
        result = Itg.integrate_newton(x, fx, 'simp1/3')
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_simp38_integral(self):
        x = np.arange(0, np.pi/2, np.pi/1000)
        fx = np.cos(x)*np.sin(x)
        result = Itg.integrate_newton(x, fx, 'simp3/8')
        self.assertAlmostEqual(result, 0.5, places=4)


if __name__ == '__main__':
    unittest.main()
