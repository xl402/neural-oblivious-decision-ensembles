import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_probability import distributions, stats
import numpy as np


@tf.function
def sparsemoid(inputs):
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
        feature_dim = input_shape[-1]
        self._build_feature_selection_logits(feature_dim)
        self._build_feature_thresholds()
        self._build_log_temperature()
        self._build_onehot_to_binary_lookup_table()
        self._build_output_response()

    def _build_feature_selection_logits(self, dim):
        initializer = tf.zeros_initializer()
        init_shape = (dim, self.n_trees, self.depth)
        init_value = initializer(shape=init_shape, dtype='float32')
        self.feature_selection_logits = tf.Variable(init_value, trainable=True)

    def _build_feature_thresholds(self):
        initializer = tf.zeros_initializer()
        init_shape = (self.n_trees, self.depth)
        init_value = initializer(shape=init_shape, dtype='float32')
        self.feature_thresholds = tf.Variable(init_value, trainable=True)

    def _build_log_temperature(self):
        initializer = tf.ones_initializer()
        init_shape = (self.n_trees, self.depth)
        init_value = initializer(shape=init_shape, dtype='float32')
        self.log_temperatures = tf.Variable(initial_value=init_value, trainable=True)

    def _build_onehot_to_binary_lookup_table(self):
        indices = tf.keras.backend.arange(0, 2 ** self.depth, 1)
        offsets = 2 ** tf.keras.backend.arange(0, self.depth, 1)
        bin_codes = (tf.reshape(indices, (1, -1)) // tf.reshape(offsets, (-1, 1)) % 2)
        bin_codes = tf.stack([bin_codes, 1 - bin_codes], axis=-1)
        bin_codes = tf.cast(bin_codes, 'float32')
        self.binary_lut = tf.Variable(initial_value=bin_codes, trainable=False)

    def _build_output_response(self):
        initializer = tf.ones_initializer()
        init_shape = (self.n_trees, self.units, 2**self.depth)
        init_value = initializer(init_shape, dtype='float32')
        self.response = tf.Variable(initial_value=init_value, trainable=True)

    def _data_aware_initialization(self, inputs):
        feature_values = self._get_feature_values(inputs)
        self._initialize_feature_thresholds(feature_values)
        self._initialize_log_temperatures(feature_values)

    def _get_feature_values(self, inputs, training=None):
        feature_selectors = tfa.activations.sparsemax(self.feature_selection_logits)
        feature_values = tf.einsum('bi,ind->bnd', inputs, feature_selectors)
        return feature_values

    def _initialize_feature_thresholds(self, inputs):
        sampler = distributions.Beta(self.threshold_init_beta, self.threshold_init_beta)
        percentiles_q = (100 * sampler.sample([self.n_trees * self.depth]))

        flattened_feature_values = tf.map_fn(tf.keras.backend.flatten, inputs)
        percentile = stats.percentile(flattened_feature_values, percentiles_q, axis=0)
        init_feature_thresholds = tf.linalg.diag_part(percentile)

        feature_thresholds = tf.reshape(init_feature_thresholds,
                                        self.feature_thresholds.shape)
        self.feature_thresholds.assign(feature_thresholds)

    def _initialize_log_temperatures(self, inputs):
        input_threshold_diff = tf.math.abs(inputs - self.feature_thresholds)
        self.log_temperatures.assign(stats.percentile(input_threshold_diff, 50, axis=0))

    def call(self, inputs, training=None):
        if not self.initialized:
            self._data_aware_initialization(inputs)
            self.initialized = True

        feature_values = self._get_feature_values(inputs)

        threshold_logits = (feature_values - self.feature_thresholds)
        threshold_logits = threshold_logits * tf.math.exp(-self.log_temperatures)
        threshold_logits = tf.stack([-threshold_logits, threshold_logits], axis=-1)

        feature_gates = sparsemoid(threshold_logits)

        # b: batch, n: number of trees, d: depth of trees, s: 2 (binary channels)
        # c: 2**depth, u: units (response units)
        response_gates = tf.einsum('bnds,dcs->bndc', feature_gates, self.binary_lut)
        response_gates = tf.math.reduce_prod(response_gates, axis=-2)
        response = tf.einsum('bnc,nuc->bnu', response_gates, self.response)
        output = tf.reduce_mean(response, axis=1)
        return output


if __name__=='__main__':
    layer = ObliviousDecisionTree(n_trees=100, depth=3, units=2)
    x = tf.random.uniform(shape=(1, 10))
    y = layer(x)
