# EDMnets
The code for "Generating valid Euclidean distance matrices" [[arXiv](https://arxiv.org/abs/1910.03131)].
If you use this code in a publication, please cite:

```text
Moritz Hoffmann, Frank Noé: Generating valid Euclidean distance matrices
arXiv:1910.03131 (2019)

bibtex:
{HoffmannNoe_EDMnets_2019,
author = {Moritz Hoffmann and Frank Noé},
title = {Generating valid Euclidean distance matrices},
journal = {arXiv:1910.03131},
year = {2019}
}
```

Requirements: Tensorflow 2.0.

Installation via

```
git clone https://github.com/noegroup/EDMnets
cd EDMnets
git submodule update --init
python setup.py install
```

#### Example usage of Hungarian reordering (also under examples directory):
```python
import numpy as np
import tensorflow as tf
import edmnets.utils as utils
import edmnets.layers as layers

# generate some data
X = np.random.normal(size=(10, 100, 5))
# associate random point types to data
types = np.random.randint(0, 10, size=(10, 100), dtype=np.int32)
# index array for shuffling
ix = np.arange(100)
Xshuffled = []
types_shuffled = []
# for each element in the batch
for i in range(len(X)):
    # shuffle the index array
    np.random.shuffle(ix)
    # append shuffled positional data and point types
    Xshuffled.append(X[i][ix])
    types_shuffled.append(types[i][ix])
Xshuffled = np.stack(Xshuffled)
types_shuffled = np.stack(types_shuffled)

# convert point to EDMs and convert to Tensorflow constants
D1 = tf.constant(utils.to_distance_matrices(X), dtype=tf.float32)
D2 = tf.constant(utils.to_distance_matrices(Xshuffled), dtype=tf.float32)

# convert types to tensorflow constants
t1 = tf.constant(types, dtype=tf.int32)
t2 = tf.constant(types_shuffled, dtype=tf.int32)

# apply reordering on D1 and t1 based on D2 and t2
D1_reordered, t1_reordered = layers.HungarianReorder()([D1, D2, t1, t2])
# check that we found the correct permutation
np.testing.assert_array_almost_equal(D1_reordered, D2)
np.testing.assert_array_almost_equal(t1_reordered, t2)
```

The output of the `HungarianReorder`-layer is equipped with a `stop_gradient`, i.e., the gradient cannot 
backpropagate through that particular part of the graph. 

#### Example usage of EDM parameterization (also under examples directory):

```python
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
        # make spd (this could also be done with expmh or cutting eigenvalues, 
        # see layers.Expmh, layers.CutEVmh).
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
```
![output](https://github.com/noegroup/EDMnets/blob/master/examples/out.png)
