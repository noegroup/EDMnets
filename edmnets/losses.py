import tensorflow as tf

import edmnets.layers as layers


def edm_loss(D, n_atoms):
    """
    Loss imposing a soft constraint on EDMness for the input matrix `D`.
    :param D: a hollow symmetric matrix
    :param n_atoms: number of atoms
    :return: loss value
    """
    # D is EDM iff T = -0.5 JDJ is positive semi-definite
    T = layers.D2T(n_atoms, name="D2J")(D)
    J_ev = tf.linalg.eigvalsh(T)
    return tf.reduce_sum(tf.square(tf.nn.relu(-J_ev)), axis=-1)

def rank_penalty(D, target_rank):
    M = layers.D2M()(D)
    eigenvalues = tf.linalg.eigvalsh(M)
    n_eigenvalues = tf.shape(eigenvalues)[1]
    undesired_evs, _ = tf.math.top_k(-eigenvalues, k=n_eigenvalues - target_rank, sorted=False,
                                     name="undesired_evs")
    undesired_evs = tf.square(undesired_evs)
    return tf.reduce_sum(undesired_evs, axis=-1)

