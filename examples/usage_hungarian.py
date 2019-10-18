import numpy as np
import tensorflow as tf
import edmnets.utils as utils
import edmnets.layers as layers


def usage_numpy():
    X = np.random.normal(size=(10, 100, 5)).astype(np.float32)
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

    D1 = utils.to_distance_matrices(X)
    D2 = utils.to_distance_matrices(Xshuffled)

    D1reordered, types_reordered = utils.reorder_distance_matrices(D1, D2, types1=types, types2=types_shuffled)
    np.testing.assert_array_almost_equal(D2, D1reordered)
    np.testing.assert_array_almost_equal(types_shuffled, types_reordered)


def usage_tensorflow():
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


if __name__ == '__main__':
    usage_numpy()
    usage_tensorflow()
