from tabnet import TabNet
from node.experiments.fraud_detection.utils import run_keras_experiment
from tensorflow.keras.layers import Dense
import tensorflow as tf


def build_tabnet(n_features):
    model = tf.keras.Sequential(
        [
            TabNet(
                feature_columns=None,
                num_features=n_features,
                feature_dim=128,
                output_dim=8,
                num_decision_steps=1,
                relaxation_factor=1.5,
                sparsity_coefficient=1e-5,
            ),
            Dense(1, activation="sigmoid")
        ]
    )
    loss = 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=[tf.keras.metrics.AUC(name='auc')])
    return model


if __name__ == "__main__":
    # train_file, test_file, num_features = 'train_base.csv', 'test_base.csv', 46
    train_file, test_file, num_features = 'train_features.csv', 'test_features.csv', 281
    run_keras_experiment(train_file, test_file, build_tabnet, args={'n_features': num_features})
