import tensorflow as tf
from tensorflow.keras.layers import Dense

from node.networks.layer import ObliviousDecisionTree as ODT
from node.networks.entmax import sparsemoid


class NODE(tf.keras.Model):
    def __init__(self,
                 units=1,
                 n_layers=1,
                 link=tf.identity,
                 n_trees=3,
                 tree_depth=4,
                 binary_selector=sparsemoid,
                 threshold_init_beta=1):

        super(NODE, self).__init__()
        self.units = units
        self.n_layers = n_layers
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.units = units
        self.threshold_init_beta = threshold_init_beta
        self.bn = tf.keras.layers.BatchNormalization()
        self.ensemble = [ODT(n_trees=n_trees,
                             depth=tree_depth,
                             units=units,
                             bin_selector=binary_selector,
                             threshold_init_beta=threshold_init_beta)
                         for _ in range(n_layers)]
        self.link = link

    def build(self, shape):
        self.feature_dense = Dense(units=shape[-1],
                                   activation='relu',
                                   kernel_initializer='identity',
                                   bias_initializer='zeros')

    def call(self, inputs, training=None):
        x = self.bn(inputs, training=training)
        x = self.feature_dense(x)
        h = 0.
        for tree in self.ensemble:
            h = h + tree(x)
            x = tf.concat([x, h], axis=1)
        return self.link(h)
