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

    def test_hungarian_without_noise(self):
        X = np.random.normal(size=(10, 100, 5))
        types = np.random.randint(0, 10, size=(10, 100))
        ix = np.arange(100)
        Xshuffled = []
        types_shuffled = []
        for i in range(len(X)):
            np.random.shuffle(ix)
            Xshuffled.append(X[i][ix])
            types_shuffled.append(types[i][ix])
        Xshuffled = np.stack(Xshuffled)
        types_shuffled = np.stack(types_shuffled)

        D1 = utils.to_distance_matrices(X)
        D2 = utils.to_distance_matrices(Xshuffled)

        D1reordered, types_reordered = utils.reorder_distance_matrices(D1, D2, types1=types, types2=types_shuffled)
        np.testing.assert_array_almost_equal(D2, D1reordered, decimal=5)
        np.testing.assert_array_almost_equal(types_shuffled, types_reordered)

    def test_hungarian_with_noise(self):
        X = np.random.normal(size=(10, 100, 5))
        ix = np.arange(100)

        noise = np.random.normal(scale=.00001, size=(100, 5))
        Xshuffled = []
        for i in range(len(X)):
            np.random.shuffle(ix)
            Xshuffled.append(X[i][ix] + noise)
        Xshuffled = np.stack(Xshuffled)

        D1 = utils.to_distance_matrices(X)
        D2 = utils.to_distance_matrices(Xshuffled)

        D1reordered, types = utils.reorder_distance_matrices(np.copy(D1), np.copy(D2))
        assert np.max(np.abs(D2 - D1reordered)) < 1e-2
        np.testing.assert_array_almost_equal(types, np.zeros_like(types))
