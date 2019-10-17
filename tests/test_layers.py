import tensorflow as tf
import numpy as np
import edmnets.layers as layers
import edmnets.utils as utils

class TestLayers(tf.test.TestCase):

    def test_expmh(self):
        L = tf.random.normal(stddev=0.1, shape=(100, 20, 20))
        L = L + tf.linalg.matrix_transpose(L)
        expL, _, exp_ev = layers.Expmh()(L)
        evs = tf.linalg.eigvalsh(expL)
        self.assertAllGreaterEqual(exp_ev, 0.)
        np.testing.assert_array_almost_equal(exp_ev, evs, decimal=5)

    def test_softplusmh(self):
        L = tf.random.normal(stddev=10, shape=(100, 20, 20))
        L = L + tf.linalg.matrix_transpose(L)
        spL, _, sp_ev = layers.Softplusmh()(L)
        evs = tf.linalg.eigvalsh(spL)
        self.assertAllGreaterEqual(sp_ev, 0.)
        np.testing.assert_array_almost_equal(sp_ev, evs, decimal=3)

    def test_D2M_M2D(self):
        X = tf.random.normal(stddev=10, shape=(100, 20, 20))
        D = layers.to_distmat(X)
        np.testing.assert_array_almost_equal(D, utils.to_distance_matrices(X), decimal=3)
        M = layers.D2M()(D)
        Mref = []
        for d in D.numpy():
            Mref.append(utils.to_M_matrix(d))
        Mref = np.stack(Mref)
        np.testing.assert_array_almost_equal(M, Mref, decimal=3)
        D2 = layers.M2D()(M)
        np.testing.assert_array_almost_equal(D2, D, decimal=3)

    def test_D2T(self):
        X = tf.random.normal(stddev=10, shape=(100, 20, 20))
        Tref = np.stack([utils.to_T_matrix(d) for d in utils.to_distance_matrices(X)])
        D = layers.to_distmat(X)
        T = layers.D2T(n_atoms=20)(D)
        np.testing.assert_array_almost_equal(T, Tref, decimal=3)

if __name__ == '__main__':
    tf.test.main()
