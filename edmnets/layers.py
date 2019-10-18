import numpy as np
import tensorflow as tf
import edmnets.utils as utils


@tf.function
def to_difference_matrix(X):
    """
    Converts a batch of coordinates to a batch of difference tensors
    :param X: batch of coordinates [B, N, d]
    :return: batch of difference tensors [B, N, N, d]
    """
    return tf.expand_dims(X, axis=2) - tf.expand_dims(X, axis=1)


@tf.function
def to_distmat(x):
    """
    Converts a batch of coordinates to a batch of EDMs
    :param x: batch of coordinates [B, N, d]
    :return: batch of EDMs [B, N, N]
    """
    diffmat = to_difference_matrix(x)
    return tf.reduce_sum(tf.square(diffmat), axis=-1, keepdims=False)


class HungarianReorder(tf.keras.layers.Layer):
    """
    Hungarian method on distance matrices. Takes as input two batches of distance matrices D1 and D2
    as well as (optionally) corresponding point types. The distance matrices and respective point types of D1
    are permuted to match D2 as closely as possible. A stop gradient is applied to the output of the reordering.
    """

    @tf.function
    def hungarian_cost(self, D1, D2):
        return tf.abs(tf.reduce_mean(D1, axis=1)[..., None] - tf.reduce_mean(D2, axis=1)[:, None, :])

    @staticmethod
    def _reorder_proxy(cost, D1, D2, types1, types2):
        return utils.reorder_distance_matrices(D1.numpy(), D2.numpy(), cost.numpy(), types1.numpy(), types2.numpy())

    @tf.function
    def call(self, inputs):
        D1, D2 = inputs[0], inputs[1]
        batch_size = tf.shape(D1)[0]
        n_atoms = D1.shape[1]
        if len(inputs) > 2:
            types1, types2 = inputs[2], inputs[3]
        else:
            types1 = tf.zeros((batch_size, n_atoms), dtype=tf.int32)
            types2 = tf.zeros((batch_size, n_atoms), dtype=tf.int32)

        cost = self.hungarian_cost(D1, D2)
        out = tf.py_function(HungarianReorder._reorder_proxy, [cost, D1, D2, types1, types2], [D1.dtype, types1.dtype])
        D1_reordered, types1_reordered = out

        D1_reordered = tf.stop_gradient(D1_reordered)
        D1_reordered = tf.cast(D1_reordered, D1.dtype)
        D1_reordered.set_shape([None, n_atoms, n_atoms])

        types1_reordered = tf.stop_gradient(types1_reordered)
        types1_reordered = tf.cast(types1_reordered, types1.dtype)
        types1_reordered.set_shape([None, n_atoms])
        return D1_reordered, types1_reordered


class Expmh(tf.keras.layers.Layer):
    """
    Matrix exponential function layer on symmetric matrix input, i.e., Expmh(H) = V Expmh(L) V^T.
    The resulting matrix is symmetric positive definite.
    """

    def call(self, inputs, **kw):
        eigenvalues, eigenvectors = tf.linalg.eigh(inputs)
        exp_ev = tf.math.exp(eigenvalues)
        exp_ev_D = tf.linalg.diag(exp_ev)
        eigenvectors_T = tf.linalg.matrix_transpose(eigenvectors)
        expA = eigenvectors @ exp_ev_D @ eigenvectors_T
        return expA, eigenvalues, exp_ev


class Softplusmh(tf.keras.layers.Layer):
    """
    Matrix softplus function for symmetric input matrices. The resulting matrix is symmetric positive definite.
    """

    def call(self, inputs, **kw):
        eigenvalues, eigenvectors = tf.linalg.eigh(inputs)
        sp_ev = tf.math.softplus(eigenvalues)
        sp_ev_D = tf.linalg.diag(sp_ev)
        eigenvectors_T = tf.linalg.matrix_transpose(eigenvectors)
        spA = eigenvectors @ sp_ev_D @ eigenvectors_T
        return spA, eigenvalues, sp_ev


class CutEVmh(tf.keras.layers.Layer):
    """
    Applies softplus to the largest ndim eigenvalues and sets the rest to zero, making the matrix positive semi-definite
    with rank `ndim`.
    """

    def __init__(self, ndim, **kw):
        super(CutEVmh, self).__init__(**kw)
        self.ndim = ndim

    def build(self, input_shape):
        self.n_particles = input_shape[1]
        self.mask = np.ones((self.n_particles,))
        for i in range(0, self.n_particles - self.ndim):
            self.mask[i] = 0.
        # print(self.mask)
        self.mask = tf.constant(self.mask, self.dtype)

    def call(self, inputs, training=None):
        eigenvalues, eigenvectors = tf.linalg.eigh(inputs)
        sp_ev = tf.math.softplus(eigenvalues)
        sp_ev = sp_ev * self.mask[None, ...]
        sp_ev_D = tf.linalg.diag(sp_ev)
        eigenvectors_T = tf.transpose(eigenvectors, perm=[0, 2, 1])
        spA = eigenvectors @ sp_ev_D @ eigenvectors_T
        return spA, eigenvalues, sp_ev


class D2M(tf.keras.layers.Layer):
    """
    Converts an EDM `D` to its Gram matrix representation `M`.
    """

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]
        n_atoms = tf.shape(inputs)[1]
        D1j = tf.reshape(tf.tile(inputs[:, 0, :], [1, n_atoms]),
                         shape=(batch_size, n_atoms, n_atoms))
        Di1 = tf.transpose(D1j, perm=[0, 2, 1])
        M = .5 * (-inputs + D1j + Di1)
        return M


class L2M(tf.keras.layers.Layer):
    """
    Converts an L matrix (shape n-1 x n-1) to an M matrix (shape n x n) by padding the first column and row with zeros.
    """

    def call(self, inputs, **kw):
        return tf.pad(inputs, [[0, 0], [1, 0], [1, 0]])


class M2D(tf.keras.layers.Layer):
    """
    Converts a Gram matrix `M` to an EDM `D`.
    """

    def call(self, inputs, **kwargs):
        M = inputs
        M_diag = tf.linalg.diag_part(M)
        return tf.expand_dims(M_diag, 1) + tf.expand_dims(M_diag, 2) - 2. * M


class D2T(tf.keras.layers.Layer):
    """
    Converts a matrix `D` to the matrix `T = -.5 J D J`. `D` is EDM iff `T` is positive semi-definite.
    """

    def __init__(self, n_atoms, **kwargs):
        self.n_atoms = n_atoms
        super().__init__(**kwargs)

    def build(self, input_shape):
        eye = tf.eye(num_rows=self.n_atoms, dtype=self.dtype)
        J = eye - tf.ones(shape=(self.n_atoms, self.n_atoms), dtype=self.dtype) / float(self.n_atoms)
        self.J = tf.reshape(J, shape=(-1, self.n_atoms, self.n_atoms), name="reshape_J")
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        J = tf.tile(self.J, multiples=[tf.shape(inputs)[0], 1, 1])
        T = -0.5 * tf.matmul(tf.matmul(J, inputs), J)
        # D is EDM iff T is positive semi-definite
        return T
