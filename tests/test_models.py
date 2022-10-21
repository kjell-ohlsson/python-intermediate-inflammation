"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])


@pytest.mark.parametrize(
    "test_mean, expected_mean",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
    ])
def test_daily_mean(test_mean, expected_mean):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test_mean)), np.array(expected_mean))

@pytest.mark.parametrize(
    "test_min, expected_min",
    [
        ([[7, 8], [9, 10], [11, 12]], [7, 8]),
        ([[1, 2], [3, 4], [5, 6]], [1, 2]),
    ])
def test_daily_min(test_min, expected_min):
    """Test mean function works."""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(np.array(test_min)), np.array(expected_min))

@pytest.mark.parametrize(
    "test_max, expected_max",
    [
        ([[7, 8], [9, 10], [11, 12]], [11, 12]),
        ([[1, 2], [3, 4], [5, 6]], [5, 6]),
    ])
def test_daily_max(test_max, expected_max):
    """Test mean function works."""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test_max)), np.array(expected_max))

@pytest.mark.parametrize(
    "test_std, expected_std",
    [
        ([[7, 8], [9, 15], [11, 20]], [1.633, 4.922]),
        ([[1, 2], [3, 7], [5, 19]], [1.633, 7.134]),
    ])
def test_daily_std(test_std, expected_std):
    """Test standard deviation function works."""
    from inflammation.models import daily_std
    npt.assert_allclose(daily_std(np.array(test_std)), np.array(expected_std), rtol=1e-04)

git