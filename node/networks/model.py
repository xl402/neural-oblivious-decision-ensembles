import tensorflow as tf

from node.networks.layer import ODST


@tf.function
def identity(x: tf.Tensor):
    return x


class NODE(tf.keras.Model):
    def __init__(self,
                 units=1,
                 n_layers=1,
                 link=tf.identity,
                 n_trees=3,
                 depth=4,
                 threshold_init_beta=1.,
                 feature_column=None):

        super(NODE, self).__init__()
        self.units = units
        self.n_layers = n_layers
        self.n_trees = n_trees
        self.depth = depth
        self.units = units
        self.threshold_init_beta = threshold_init_beta
        self.feature_column = feature_column

        if feature_column is None:
            self.feature = tf.keras.layers.Lambda(identity)
        else:
            self.feature = feature_column
        self.bn = tf.keras.layers.BatchNormalization()
        self.ensemble = [ODST(n_trees=n_trees,
                              depth=depth,
                              units=units,
                              threshold_init_beta=threshold_init_beta)
                         for _ in range(n_layers)]
        self.link = link

    def call(self, inputs, training=None):
        X = self.feature(inputs)
        X = self.bn(X, training=training)
        for tree in self.ensemble:
            H = tree(X)
            X = tf.concat([X, H], axis=1)
        return self.link(H)
