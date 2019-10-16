import tensorflow as tf


class Expmh(tf.keras.layers.Layer):
    """
    Matrix exponential function layer on symmetric matrix input, i.e., Expmh(H) = V Expmh(L) V^T.
    The resulting matrix is symmetric positive definite.
    """

    def call(self, inputs, **kw):
        eigenvalues, eigenvectors = tf.linalg.eigh(inputs)
        exp_ev = tf.math.exp(eigenvalues)
        exp_ev_D = tf.linalg.diag(exp_ev)
        eigenvectors_T = tf.transpose(eigenvectors, perm=[0, 2, 1])
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
