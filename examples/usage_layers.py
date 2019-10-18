import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm

import edmnets.fcnn as fcnn
import edmnets.layers as layers
import edmnets.losses as losses
import edmnets.utils as utils


class EDMLayer(tf.keras.layers.Layer):

    def __init__(self, **kw):
        super(EDMLayer, self).__init__(**kw)

        self.mlp = fcnn.FCNNLayer(units=[20, 50, 70 * 70])
        self.l2m = layers.L2M()
        self.m2d = layers.M2D()
        self.spmh = layers.Softplusmh()

    def __call__(self, inputs, training=None):
        # run inputs through multilayer preceptron
        output = self.mlp(inputs, training=training)
        # reshape into square matrices
        output = tf.reshape(output, (-1, 70, 70))
        # symmetrize
        L = .5 * (output + tf.linalg.matrix_transpose(output))
        # make spd (this could also be done with expmh or cutting eigenvalues, see layers.Expmh, layers.CutEVmh).
        L, _, ev_L = self.spmh(L)
        # convert to Gram matrix
        M = self.l2m(L)
        # convert to EDM
        D = self.m2d(M)
        return D, ev_L


# create a layer that outputs something which can be trained to be a EDM
net = EDMLayer()
# some optimizer
optimizer = tf.keras.optimizers.Adam(1e-1)


@tf.function
def train_step():
    # training step which first creates some white noise...
    z = tf.random.normal(shape=(64, 15))
    with tf.GradientTape() as tape:
        # ... and produces (quasi) EDMs and eigenvalues
        D, ev = net(z, training=True)
        # check that -0.5 JDJ is positive semi-definite
        loss_edm = losses.edm_loss(D, n_atoms=71)
        # want embedding dimension 2
        loss_rank = losses.rank_penalty(D, target_rank=2)
        # loss is just the sum of the two
        loss = loss_edm + loss_rank
    # evaluate and apply gradients
    gradients = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, net.trainable_variables))
    return loss


# training loop
loss_values = []
for step in tqdm.tqdm(range(1000)):
    loss = train_step()
    loss_values.append(np.mean(loss))

# evaluate the net on some random samples
D, _ = net(tf.random.normal(shape=(64, 15)), training=False)
# convert to two-dimensional coordinates
X = utils.to_coordinates(D.numpy(), squared=True, ndim=2)

# plot loss curve and scatter of the first three configurations
f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

ax1.semilogy(np.array(loss_values), label='rank penalty')
ax1.set_xlabel('step')
ax1.set_ylabel('loss')
ax1.legend()
ax2.scatter(X[0, :, 0], X[0, :, 1], label='first batch element')
ax2.scatter(X[1, :, 0], X[1, :, 1], label='second batch element')
ax2.scatter(X[2, :, 0], X[2, :, 1], label='third batch element')
ax2.legend()
ax2.set_xlabel('x')
ax2.set_ylabel('y')
plt.show()
