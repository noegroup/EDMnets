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

    def test_hungarian_reorder(self):
        X = np.random.normal(size=(10, 100, 5))
        types = np.random.randint(0, 10, size=(10, 100), dtype=np.int32)
        ix = np.arange(100)
        Xshuffled = []
        types_shuffled = []
        for i in range(len(X)):
            np.random.shuffle(ix)
            Xshuffled.append(X[i][ix])
            types_shuffled.append(types[i][ix])
        Xshuffled = np.stack(Xshuffled)
        types_shuffled = np.stack(types_shuffled)

        D1 = tf.constant(utils.to_distance_matrices(X), dtype=tf.float32)
        D2 = tf.constant(utils.to_distance_matrices(Xshuffled), dtype=tf.float32)

        t1 = tf.constant(types, dtype=tf.int32)
        t2 = tf.constant(types_shuffled, dtype=tf.int32)

        D1_reordered, t1_reordered = layers.HungarianReorder()([D1, D2, t1, t2])
        np.testing.assert_array_almost_equal(D1_reordered, D2)
        np.testing.assert_array_almost_equal(t1_reordered, t2)

    def test_hungarian_reorder_notypes(self):
        X = np.random.normal(size=(10, 100, 5))
        ix = np.arange(100)
        Xshuffled = []
        for i in range(len(X)):
            np.random.shuffle(ix)
            Xshuffled.append(X[i][ix])
        Xshuffled = np.stack(Xshuffled)

        D1 = tf.constant(utils.to_distance_matrices(X), dtype=tf.float32)
        D2 = tf.constant(utils.to_distance_matrices(Xshuffled), dtype=tf.float32)

        D1_reordered, t1_reordered = layers.HungarianReorder()([D1, D2])
        np.testing.assert_array_almost_equal(D1_reordered, D2, decimal=5)


if __name__ == '__main__':
    tf.test.main()
