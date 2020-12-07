import tensorflow as tf
import numpy as np
import pytest
from tensorflow.keras.models import load_model

from node.networks.model import NODE


def test_node_can_predict():
    model = NODE(units=2)
    x = tf.random.uniform((5, 100), dtype='float32')
    y = model(x)
    assert all(np.array(y.shape)==(5, 2))


def test_can_save_as_tensorflow_model(tmpdir):
    model, (x, y) = get_fitted_model(n_layers=3)
    model.save(tmpdir.join('model'))
    reconstructed_model = load_model(tmpdir.join('model'))
    result = model.predict(x)
    reconstructed_result = reconstructed_model.predict(x)
    assert np.allclose(result, reconstructed_result)


def get_fitted_model(**kwargs):
    model = NODE(**kwargs)
    x = tf.random.uniform(shape=(1, 4), dtype='float32')
    y = tf.random.uniform(shape=(1, 1), dtype='float32')
    inputs = tf.keras.layers.Input(shape=[4])
    outputs = model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile("adam", "mean_squared_error")
    model.fit(x, y, epochs=1)
    fitting_data = (x, y)
    return model, fitting_data
