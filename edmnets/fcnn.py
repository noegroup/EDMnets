from typing import List, Union

import tensorflow as tf


class FCNNLayer(tf.keras.layers.Layer):

    def __init__(self, units, use_bias: bool = True, initializer=None,
                 batch_normalization: Union[bool, List[int]] = False, dropout: float = 0.0,
                 activation=tf.keras.layers.LeakyReLU,
                 output_activation=None, output_dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.initializer = initializer if initializer is not None else tf.keras.initializers.he_normal
        self.batch_normlization = batch_normalization
        self.dropout = dropout
        self.activation = activation
        self.output_activation = output_activation
        self.output_dropout = output_dropout
        for ix, n_neurons in enumerate(units[:-1]):
            setattr(self, f"L{ix}", tf.keras.layers.Dense(units=n_neurons, activation=None,
                                                          use_bias=self.use_bias, name=f"L{ix}",
                                                          kernel_initializer=self.initializer()))
            if (isinstance(self.batch_normlization, bool) and self.batch_normlization is True) or \
                    (isinstance(self.batch_normlization, (tuple, list)) and ix in self.batch_normlization):
                setattr(self, f"L{ix}_bn", tf.keras.layers.BatchNormalization(name=f"L{ix}_bn", axis=-1))
            if self.activation is not None:
                setattr(self, f"L{ix}_activation", self.activation(name=f"L{ix}_activation"))
            if self.dropout > 0:
                setattr(self, f"L{ix}_dropout", tf.keras.layers.Dropout(name=f"L{ix}_dropout", rate=self.dropout))
        setattr(self, f"L{len(units) - 1}",
                tf.keras.layers.Dense(units=units[-1], activation=None, use_bias=use_bias,
                                      name=f"L{len(units) - 1}", kernel_initializer=self.initializer()))
        if output_activation is not None:
            setattr(self, f"L{len(units) - 1}_activation", output_activation(name=f"L{len(units) - 1}_activation"))
        if output_dropout > 0:
            setattr(self, f"L{len(units) - 1}_dropout", tf.keras.layers.Dropout(name=f"L{len(units) - 1}_dropout",
                                                                                rate=output_dropout))

    def call(self, inputs, training=None):
        out = inputs
        for ix in range(len(self.units)):
            if hasattr(self, f"L{ix}"):
                out = getattr(self, f"L{ix}")(out)
            if hasattr(self, f"L{ix}_bn"):
                out = getattr(self, f"L{ix}_bn")(out, training=training)
            if hasattr(self, f"L{ix}_activation"):
                out = getattr(self, f"L{ix}_activation")(out)
            if hasattr(self, f"L{ix}_dropout"):
                out = getattr(self, f"L{ix}_dropout")(out, training=training)
        return out
