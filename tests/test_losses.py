import tensorflow as tf
import numpy as np
import edmnets.layers as layers
import edmnets.losses as losses

class TestLayers(tf.test.TestCase):

    def test_edm_loss(self):
        edms = layers.to_distmat(tf.random.normal(stddev=10, shape=(100, 20, 3)))
        l = losses.edm_loss(edms, n_atoms=20)
        # loss is == 0
        np.testing.assert_array_almost_equal(l, np.zeros_like(l))
        not_edms = tf.random.uniform(minval=0, maxval=1, shape=(100, 20, 20))
        not_edms = not_edms + tf.linalg.matrix_transpose(not_edms)
        l = losses.edm_loss(not_edms, n_atoms=20)
        # loss is > 0
        np.testing.assert_array_less(np.zeros_like(l), l)

    def test_rank_penalty(self):
        ndim = 5
        edms = layers.to_distmat(tf.random.normal(stddev=10, shape=(100, 20, ndim)))
        l = losses.rank_penalty(edms, ndim)
        np.testing.assert_array_almost_equal(l, np.zeros_like(l), decimal=4)
        l = losses.rank_penalty(edms, 3)
        np.testing.assert_array_less(np.zeros_like(l), l)


if __name__ == '__main__':
    tf.test.main()
