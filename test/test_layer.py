import tensorflow as tf
import numpy as np
import pytest

from node.networks.layer import get_binary_lookup_table

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
