import unittest
import numpy as np
from goph420_lab01 import integration as Itg


class TestNewton(unittest.TestCase):

    def test_trap_integral(self):
        x = np.arange(0, 10, 0.0000001)
        fx = 3*x + 2
        result = Itg.integrate_newton(x, fx, 'trap')
        self.assertAlmostEqual(result, 170, places=4)

        x = np.arange(0, 2, 0.000001)
        fx = (x + 2)
        result = Itg.integrate_newton(x, fx, 'trap')
        self.assertAlmostEqual(result, 6, places=4)

        x = np.arange(-1, 1, 0.000001)
        fx = (3*x**3 - x**2 + x - 1)
        result = Itg.integrate_newton(x, fx, 'trap')
        self.assertAlmostEqual(result, -8/3, places=4)

    def test_simp13_integral(self):
        x = np.arange(0, np.pi/2, np.pi/1000)
        fx = np.cos(x)*np.sin(x)
        result = Itg.integrate_newton(x, fx, 'simp')
        self.assertAlmostEqual(result, 0.5, places=4)

        # Test with even number of data
        x = np.arange(0, 2, 0.000002)
        fx = (x**2 + 2)
        result = Itg.integrate_newton(x, fx, 'simp')
        self.assertAlmostEqual(result, 20/3, places=4)

        # Test with odd number of data
        x = np.arange(-1, 1, 0.000003)
        fx = (3*x**2 - x - 1)
        result = Itg.integrate_newton(x, fx, 'simp')
        self.assertAlmostEqual(result, 0, places=4)

    def test_raises_newton(self):
        # Test raises:
        x = [1, 2, 3]
        fx = [1, 2, 3]
        # Raise for alg different from listed ones:
        self.assertRaises(ValueError, Itg.integrate_newton, x, fx, 'trapo')
        # Raise for incompatible size of x and fx:
        self.assertRaises(ValueError, Itg.integrate_newton,
                          x[0:1], fx)
        # Raise in simp, for simp are requiered at least 3 points.
        self.assertRaises(ValueError, Itg.integrate_newton,
                          [1, 2], [1, 2], 'simp')


if __name__ == '__main__':
    unittest.main()
