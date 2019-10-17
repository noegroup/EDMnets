import unittest

import numpy as np

import edmnets.utils as utils
from scipy.spatial import distance_matrix

class TestUtils(unittest.TestCase):

    def test_is_edm(self):
        X = np.random.normal(size=(10, 100, 5))
        D = utils.to_distance_matrices(X, squared=True)
        assert np.all(utils.is_edm(D))

    def test_to_distance_matrices(self):
        X = np.random.normal(size=(10, 100, 5))
        D = utils.to_distance_matrices(X, squared=True)
        Ds = []
        for x in X:
            Ds.append(np.square(distance_matrix(x, x)))
        np.testing.assert_array_almost_equal(D, np.stack(Ds))

    def test_to_M_to_D(self):
        X = np.random.normal(size=(1, 100, 5))
        D = utils.to_distance_matrices(X, squared=True)[0]
        M = utils.to_M_matrix(D)
        np.testing.assert_array_almost_equal(M[:, 0], np.zeros((100,)))
        np.testing.assert_array_almost_equal(M[0, :], np.zeros((100,)))
        Drec = utils.to_D_matrix(M)
        np.testing.assert_array_almost_equal(D, Drec)

    def test_to_coordinates(self):
        X = np.random.normal(size=(10, 100, 5))
        D = utils.to_distance_matrices(X)
        Xrec = utils.to_coordinates(D, squared=True, ndim=5)
        assert Xrec.shape == X.shape
        Drec = utils.to_distance_matrices(Xrec)
        np.testing.assert_array_almost_equal(D, Drec)
