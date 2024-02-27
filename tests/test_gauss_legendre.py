import unittest
from goph420_lab01 import integration as Itg


class TestGaussLegendre(unittest.TestCase):

    def test_Gauss(self):
        sigma = 1
        lims = [0, 2.383*sigma]
        npts = 2
        fx = Itg.GaussDist()
        result = Itg.integrate_gauss(fx, lims, npts)
        self.assertAlmostEqual(2*result, 1, places=4)

        lims = [0, 4.335*sigma]
        npts = 3
        fx = Itg.GaussDist()
        result = Itg.integrate_gauss(fx, lims, npts)
        self.assertAlmostEqual(2*result, 1, places=4)

        lims = [0, 7.498*sigma]
        npts = 4
        fx = Itg.GaussDist()
        result = Itg.integrate_gauss(fx, lims, npts)
        self.assertAlmostEqual(2*result, 1, places=4)

    def test_raises_gauss(self):
        # Test raises:
        f = Itg.GaussDist()

        lims = [0]
        npts = 3
        self.assertRaises(ValueError, Itg.integrate_gauss, f, lims, npts)

        lims = [0, 1]
        npts = 1
        self.assertRaises(ValueError, Itg.integrate_gauss, f, lims, npts)

        lims = ['0', '1']
        npts = 3
        self.assertRaises(TypeError, Itg.integrate_gauss, f, lims, npts)

        lims = [0, 1]
        npts = 3
        f = 'Hola'
        self.assertRaises(ValueError, Itg.integrate_gauss, f, lims, npts)


if __name__ == '__main__':
    unittest.main()
