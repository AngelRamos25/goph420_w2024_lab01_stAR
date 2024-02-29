import unittest
from goph420_lab01 import integration as Itg


class TestGaussLegendre(unittest.TestCase):

    def test_Gauss(self):
        lims = [0, 10]
        npts = 1
        fx = Itg.Polyn(npts)
        result = Itg.integrate_gauss(fx, lims, npts)
        self.assertEqual(result, 60)

        npts = 2
        fx = Itg.Polyn(npts)
        result = Itg.integrate_gauss(fx, lims, npts)
        self.assertAlmostEqual(result, 8680/3, places=9)

        npts = 3
        fx = Itg.Polyn(npts)
        result = Itg.integrate_gauss(fx, lims, npts)
        self.assertAlmostEqual(result, 189560, places=9)

        npts = 4
        fx = Itg.Polyn(npts)
        result = Itg.integrate_gauss(fx, lims, npts)
        self.assertAlmostEqual(result, 98826920/7, places=7)

        npts = 5
        fx = Itg.Polyn(npts)
        result = Itg.integrate_gauss(fx, lims, npts)
        self.assertEqual(result, 70889442280/63)

    def test_raises_gauss(self):
        # Test raises:
        f = Itg.GaussDist()

        # Raise for len(lims) != 2
        lims = [0]
        npts = 3
        self.assertRaises(ValueError, Itg.integrate_gauss, f, lims, npts)

        # Raise for npts not in [1,2,3,4,5]
        lims = [0, 1]
        npts = 6
        self.assertRaises(ValueError, Itg.integrate_gauss, f, lims, npts)

        # Raise for [a,b] are not convertible to float
        lims = ['0', '1']
        npts = 3
        self.assertRaises(TypeError, Itg.integrate_gauss, f, lims, npts)

        # Raise for f not calleable
        lims = [0, 1]
        npts = 3
        f = 'Hola'
        self.assertRaises(TypeError, Itg.integrate_gauss, f, lims, npts)


if __name__ == '__main__':
    unittest.main()
