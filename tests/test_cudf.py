"""Tests for cuDF functionality."""
import numpy as np
import pytest
import cudf


def test_series_creation():
    s = cudf.Series([1, 2, 3])
    assert len(s) == 3


def test_series_sum():
    s = cudf.Series([1, 2, 3, 4, 5])
    assert s.sum() == 15


def test_series_mean():
    s = cudf.Series([10, 20, 30])
    assert s.mean() == 20.0


def test_dataframe_creation():
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.shape == (3, 2)


def test_dataframe_filter():
    df = cudf.DataFrame({"x": list(range(10))})
    filtered = df[df["x"] > 5]
    assert len(filtered) == 4


def test_numpy_interop():
    arr = np.array([1.0, 2.0, 3.0])
    s = cudf.Series(arr)
    result = s.to_numpy()
    np.testing.assert_array_equal(result, arr)


def test_groupby():
    df = cudf.DataFrame({
        "group": ["a", "b", "a", "b"],
        "value": [1, 2, 3, 4]
    })
    result = df.groupby("group")["value"].sum().to_pandas()
    assert result["a"] == 4
    assert result["b"] == 6

