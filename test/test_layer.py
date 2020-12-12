import tensorflow as tf
import numpy as np
import pytest
from mock import Mock, patch

from node.networks.layer import get_binary_lookup_table, get_feature_selection_logits
from node.networks.layer import get_log_temperatures, get_output_response
from node.networks.layer import init_log_temperatures, init_feature_thresholds

TREE_DEPTHS = [1, 2, 3]


@pytest.mark.parametrize('depth', TREE_DEPTHS)
def test_binary_lookup_table_last_dimension_is_complementary(depth):
    lut = get_binary_lookup_table(depth).numpy()
    out = np.sum(lut, axis=-1)
    expected_shape = (depth, 2**depth)
    assert np.all(np.ones(expected_shape) == out)


@pytest.mark.parametrize('depth', TREE_DEPTHS)
def test_binary_lookup_table_enumerates_over_all_binary_combination_space(depth):
    lut = get_binary_lookup_table(depth).numpy()
    lut = lut[:, :, 0]
    expected_shape = (depth, 2**depth)
    assert {0., 1.} == set(lut.flatten())
    assert np.all(expected_shape == np.unique(lut, axis=1).shape)


def test_binary_lookup_table_returns_nontrainable_constant():
    out = get_binary_lookup_table(2)
    assert not out.trainable


def test_get_feature_selection_logits_returns_correct_shape():
    n_trees, depth, dim = 1, 3, 5
    out = get_feature_selection_logits(n_trees, depth, dim)
    assert np.all(out.shape == np.array([dim, n_trees, depth]))


def test_get_feature_selection_logits_is_trainable():
    n_trees, depth, dim = 1, 3, 5
    out = get_feature_selection_logits(n_trees, depth, dim)
    assert out.trainable


def test_get_log_temperatures_returns_correct_shape():
    n_trees, depth = 1, 3
    out = get_log_temperatures(n_trees, depth)
    assert np.all(out.shape == np.array([n_trees, depth]))


def test_get_log_temperatures_is_trainable():
    n_trees, depth = 1, 3
    out = get_log_temperatures(n_trees, depth)
    assert out.trainable


def test_get_output_response_returns_correct_shape():
    n_trees, depth, units = 1, 3, 2
    out = get_output_response(n_trees, depth, units)
    assert np.all(out.shape == np.array([n_trees, units, 2**depth]))


def test_output_response_is_trainable():
    n_trees, depth, units = 1, 3, 2
    out = get_output_response(n_trees, depth, units)
    assert out.trainable


def test_init_log_temperatures_returns_50th_percentile_value():
    features = np.random.uniform(size=(1, 10))
    feature_thresholds = np.zeros(features.shape)
    initial_log_temperature = init_log_temperatures(features, feature_thresholds)
    assert np.all(initial_log_temperature == np.percentile(features, 50, axis=0))


@patch('node.networks.layer.distributions.Beta.sample')
def test_init_feature_thresholds(percentile_q):
    n_trees, depth = 1, 3
    percentile_q.return_value = np.array([.1, .2, .3])
    features = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    out = init_feature_thresholds(features, 1, n_trees, depth).numpy()
    assert np.all(out.flatten() == [2, 3, 4])
