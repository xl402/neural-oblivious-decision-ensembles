from node.networks.model import NODE
from tensorflow.keras.layers import Dense

import tensorflow as tf
from node.experiments.fraud_detection.utils import run_keras_experiment


def build_node():
    model = tf.keras.Sequential(
        [
            NODE(units=1,
                 n_layers=3,
                 n_trees=10,
                 tree_depth=6,
                 link=tf.keras.activations.sigmoid),
        ]
    )
    loss = 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=[tf.keras.metrics.AUC()])
    return model


if __name__ == "__main__":
    # train_file, test_file = 'train_base.csv', 'test_base.csv'
    train_file, test_file = 'train_features.csv', 'test_features.csv'
    run_keras_experiment(train_file, test_file, build_node, patience=15)
