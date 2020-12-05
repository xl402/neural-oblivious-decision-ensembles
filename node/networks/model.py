import tensorflow as tf

from node.networks.layer import ObliviousDecisionTree as ODT


class NODE(tf.keras.Model):
    def __init__(self,
                 units=1,
                 n_layers=1,
                 link=tf.identity,
                 n_trees=3,
                 depth=4,
                 threshold_init_beta=1):

        super(NODE, self).__init__()
        self.units = units
        self.n_layers = n_layers
        self.n_trees = n_trees
        self.depth = depth
        self.units = units
        self.threshold_init_beta = threshold_init_beta

        self.bn = tf.keras.layers.BatchNormalization()
        self.ensemble = [ODT(n_trees=n_trees,
                             depth=depth,
                             units=units,
                             threshold_init_beta=threshold_init_beta)
                         for _ in range(n_layers)]
        self.link = link

    @tf.function
    def _identity(self, x):
        return x

    def call(self, inputs, training=None):
        x = self.bn(inputs, training=training)
        for tree in self.ensemble:
            h = tree(x)
            x = tf.concat([x, h], axis=1)
        return self.link(h)
