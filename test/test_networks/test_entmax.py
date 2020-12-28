import numpy as np
import pytest
import tensorflow as tf

from node.networks.entmax import entmax15, entmax_threshold_and_support


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_entmax15_sums_to_one_along_arbituary_axis(axis):
    x = tf.random.uniform((1, 5, 10))
    z = entmax15(x, axis=axis).numpy()
    sum_over_axis = np.sum(z, axis=axis)
    assert np.allclose(sum_over_axis, np.ones_like(sum_over_axis))


def test_entmax15_gets_correct_custom_gradient():
    x = tf.random.uniform((2, 10))
    expected, actual = tf.test.compute_gradient(entmax15, [x])
    assert np.allclose(expected[0], actual[0], atol=1e-3)
