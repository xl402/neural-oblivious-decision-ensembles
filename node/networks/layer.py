from typing import Union, Optional
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import numpy as np


@tf.function
def sparsemoid(inputs: tf.Tensor):
    return tf.clip_by_value(0.5 * inputs + 0.5, 0., 1.)


class ObliviousDecisionTree(tf.keras.layers.Layer):
    def __init__(self,
                 n_trees=3,
                 depth=4,
                 units=1,
                 threshold_init_beta=1.):
        super(ObliviousDecisionTree, self).__init__()
        self.initialized = False
        self.n_trees = n_trees
        self.depth = depth
        self.units = units
        self.threshold_init_beta = threshold_init_beta

    def build(self, input_shape):
        feature_selection_logits_init = tf.zeros_initializer()
        logits_init_shape = (input_shape[-1], self.n_trees, self.depth)
        logits_init_value = feature_selection_logits_init(shape=logits_init_shape,
                                                          dtype='float32')
        self.feature_selection_logits = tf.Variable(logits_init_value, trainable=True)

        feature_thresholds_init = tf.zeros_initializer()
        thresholds_init_shape = (self.n_trees, self.depth)
        thresholds_init_value = feature_thresholds_init(shape=thresholds_init_shape,
                                                        dtype='float32')
        self.feature_thresholds = tf.Variable(thresholds_init_value, trainable=True)

        log_temperatures_init = tf.ones_initializer()
        log_temperatures_init_value = log_temperatures_init(shape=thresholds_init_shape,
                                                            dtype='float32')
        self.log_temperatures = tf.Variable(initial_value=log_temperatures_init_value,
                                            trainable=True)

        indices = tf.keras.backend.arange(0, 2 ** self.depth, 1)
        offsets = 2 ** tf.keras.backend.arange(0, self.depth, 1)
        bin_codes = (tf.reshape(indices, (1, -1)) // tf.reshape(offsets, (-1, 1)) % 2)
        bin_codes_1hot = tf.stack([bin_codes, 1 - bin_codes], axis=-1)
        bin_codes_1hot = tf.cast(bin_codes_1hot, 'float32')
        self.bin_codes_1hot = tf.Variable(initial_value=bin_codes_1hot,
                                          trainable=False)

        response_init = tf.ones_initializer()
        response_shape = (self.n_trees, self.units, 2**self.depth)
        response_init_value = response_init(response_shape, dtype='float32')
        self.response = tf.Variable(initial_value=response_init_value,
                                    trainable=True)

    def initialize(self, inputs):
        feature_values = self.feature_values(inputs)

        # intialize feature_thresholds
        percentiles_q = (100 * tfp.distributions.Beta(self.threshold_init_beta,
                                                      self.threshold_init_beta)
                         .sample([self.n_trees * self.depth]))
        flattened_feature_values = tf.map_fn(tf.keras.backend.flatten, feature_values)
        init_feature_thresholds = tf.linalg.diag_part(tfp.stats.percentile(flattened_feature_values, percentiles_q, axis=0))

        self.feature_thresholds.assign(tf.reshape(init_feature_thresholds, self.feature_thresholds.shape))

        # intialize log_temperatures
        self.log_temperatures.assign(tfp.stats.percentile(tf.math.abs(feature_values - self.feature_thresholds), 50, axis=0))

    def feature_values(self, inputs, training=None):
        feature_selectors = tfa.activations.sparsemax(self.feature_selection_logits)
        # ^--[in_features, n_trees, depth]

        feature_values = tf.einsum('bi,ind->bnd', inputs, feature_selectors)
        # ^--[batch_size, n_trees, depth]
        return feature_values

    def call(self, inputs, training=None):
        if not self.initialized:
            self.initialize(inputs)
            self.initialized = True

        feature_values = self.feature_values(inputs)
        threshold_logits = (feature_values - self.feature_thresholds) * tf.math.exp(-self.log_temperatures)

        threshold_logits = tf.stack([-threshold_logits, threshold_logits], axis=-1)
        # ^--[batch_size, n_trees, depth, 2]

        bins = sparsemoid(threshold_logits)
        # ^--[batch_size, n_trees, depth, 2], approximately binary

        bin_matches = tf.einsum('btds,dcs->btdc', bins, self.bin_codes_1hot)
        # ^--[batch_size, n_trees, depth, 2 ** depth]

        response_weights = tf.math.reduce_prod(bin_matches, axis=-2)
        # ^-- [batch_size, n_trees, 2 ** depth]

        response = tf.einsum('bnd,ncd->bnc', response_weights, self.response)
        # ^-- [batch_size, n_trees, units]
        return tf.reduce_sum(response, axis=1)
