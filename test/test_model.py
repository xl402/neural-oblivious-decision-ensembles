import tensorflow as tf
import numpy as np
import pytest

from node.networks.model import NODE


def test_node_can_predict():
    model = NODE(units=2)
    x = tf.random.uniform((5, 100), dtype='float32')
    y = model(x)
    assert all(np.array(y.shape)==(5, 2))
