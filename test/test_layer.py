import tensorflow as tf
import numpy as np
import pytest

from node.networks.layer import get_binary_lookup_table, get_feature_selection_logits
from node.networks.layer import get_log_temperature, get_output_response

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


def test_get_log_temperature_returns_correct_shape():
    n_trees, depth = 1, 3
    out = get_log_temperature(n_trees, depth)
    assert np.all(out.shape == np.array([n_trees, depth]))


def test_get_log_temperature_is_trainable():
    n_trees, depth = 1, 3
    out = get_log_temperature(n_trees, depth)
    assert out.trainable


def test_get_output_response_returns_correct_shape():
    n_trees, depth, units = 1, 3, 2
    out = get_output_response(n_trees, depth, units)
    assert np.all(out.shape == np.array([n_trees, units, 2**depth]))


def test_output_response_is_trainable():
    n_trees, depth, units = 1, 3, 2
    out = get_output_response(n_trees, depth, units)
    assert out.trainable
