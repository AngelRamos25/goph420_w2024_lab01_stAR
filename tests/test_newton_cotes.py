import unittest
import numpy as np
from goph420_lab01 import integration as Itg


class TestTrapezoid(unittest.TestCase):

    def test_trap_integral(self):
        x = np.arange(0, np.pi/2, np.pi/1000)
        fx = np.cos(x)*np.sin(x)
        result = Itg.integrate_newton(x, fx, 'trap')
        self.assertAlmostEqual(result, 0.5, places=4)
