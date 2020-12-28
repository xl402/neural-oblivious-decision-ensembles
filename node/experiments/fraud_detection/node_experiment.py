from node.networks.model import NODE
from node.experiments.fraud_detection.utils import run_keras_experiment
from tensorflow.keras.layers import Dense
import tensorflow as tf


def build_node():
    model = tf.keras.Sequential(
        [
            NODE(units=10, n_layers=4, n_trees=10, tree_depth=6),
            Dense(1, activation="sigmoid")
        ]
    )
    loss = 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=[tf.keras.metrics.AUC()])
    return model


if __name__ == "__main__":
    # train_file, test_file = 'train_base.csv', 'test_base.csv'
    train_file, test_file = 'train_features.csv', 'test_features.csv'
    run_keras_experiment(train_file, test_file, build_node)
