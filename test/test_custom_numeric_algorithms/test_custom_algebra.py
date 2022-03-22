import unittest
import numpy as np
import custom_numeric_algorithms.custom_algebra as ca


class MyTestCase(unittest.TestCase):
    def test_L2(self):
        # Point
        point1 = np.array([2., 0.])
        self.assertEqual(ca.L2(point1), 2.0)

        # Points
        points = np.array([[2., 2.], [0, 1.], [-4., -3.], [-4., 3], [3, -4.]])
        result = np.array([np.sqrt(8.), 1., 5., 5., 5.])
        for index in np.arange(result.shape[0]):
            self.assertEqual(ca.L2(points[index]), result[index])

    def test_c_dotprod(self):
        vec_1 = np.array([1., 2.], dtype=complex)
        vec_2 = np.array([2., 1.], dtype=complex)
        self.assertEqual(ca.c_dotprod(vec_1, vec_2), 4. + 0j)


if __name__ == '__main__':
    unittest.main()
