"""Tests for cuDF functionality."""
import os
import numpy as np
import cudf
from cudf_demo import read_covid_csv, process_covid_data


def test_series_creation():
    """Series can be created with expected length."""
    s = cudf.Series([1, 2, 3])
    assert len(s) == 3


def test_series_sum():
    """Series sum returns the expected scalar value."""
    s = cudf.Series([1, 2, 3, 4, 5])
    assert s.sum() == 15


def test_series_mean():
    """Series mean returns the expected floating-point value."""
    s = cudf.Series([10, 20, 30])
    assert s.mean() == 20.0


def test_dataframe_creation():
    """DataFrame shape matches rows and columns."""
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.shape == (3, 2)


def test_dataframe_filter():
    """Filtering keeps only rows matching the predicate."""
    df = cudf.DataFrame({"x": list(range(10))})
    filtered = df[df["x"] > 5]
    assert len(filtered) == 4


def test_numpy_interop():
    """Series converts back to NumPy without data changes."""
    arr = np.array([1.0, 2.0, 3.0])
    s = cudf.Series(arr)
    result = s.to_numpy()
    np.testing.assert_array_equal(result, arr)


def test_groupby():
    """Groupby sum aggregates values by key."""
    df = cudf.DataFrame({
        "group": ["a", "b", "a", "b"],
        "value": [1, 2, 3, 4]
    })
    result = df.groupby("group")["value"].sum().to_pandas()
    assert result["a"] == 4
    assert result["b"] == 6


_COVID_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "covid_data.csv")


def test_read_covid_csv():
    """CSV file loads into a DataFrame with expected columns and rows."""
    df = read_covid_csv(_COVID_CSV)
    assert len(df) > 0, "DataFrame must not be empty"
    for col in ("continent", "location", "date", "total_cases",
                "new_cases", "total_deaths", "new_deaths"):
        assert col in df.columns, f"Expected column '{col}' not found"


def test_process_covid_data():
    """Processing aggregates data into one row per continent."""
    df = read_covid_csv(_COVID_CSV)
    summary = process_covid_data(df)
    expected_continents = {
        "Africa", "Asia", "Europe",
        "North America", "Oceania", "South America",
    }
    result_continents = set(summary["continent"].to_pandas().tolist())
    assert expected_continents == result_continents
    assert (summary["new_cases"] >= 0).all()
    assert (summary["new_deaths"] >= 0).all()
