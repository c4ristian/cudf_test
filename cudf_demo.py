"""Demo script to test cuDF on GPU."""
import os

# Use KvikIO compatible mode so cuFile/GDS is never initialised on systems
# that do not have GPUDirect Storage installed.
os.environ.setdefault("KVIKIO_COMPAT_MODE", "ON")

# pylint: disable=wrong-import-position
# Reason: KVIKIO_COMPAT_MODE must be set before cudf/cupy/numpy are imported
# because KvikIO reads the variable at import time to decide whether to use cuFile.
import cudf
import cupy as cp
import numpy as np
# pylint: enable=wrong-import-position
def try_series():
    """Test basic cuDF Series operations."""
    s = cudf.Series([1, 2, 3, 4, 5])
    print("cuDF Series:")
    print(s)
    print(f"Sum: {s.sum()}, Mean: {s.mean()}")


def try_dataframe():
    """Test basic cuDF DataFrame operations."""
    df = cudf.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "score": [88.5, 92.0, 78.3]
    })
    print("\ncuDF DataFrame:")
    print(df)
    print(f"\nMean age: {df['age'].mean()}")
    print(f"Max score: {df['score'].max()}")


def try_numpy_interop():
    """Test cuDF interoperability with NumPy."""
    arr = np.array([10, 20, 30, 40, 50])
    s = cudf.Series(arr)
    print("\ncuDF Series from NumPy array:")
    print(s)
    back_to_numpy = s.to_numpy()
    print(f"Back to NumPy: {back_to_numpy}")


def try_filter():
    """Test filtering a cuDF DataFrame."""
    df = cudf.DataFrame({
        "x": list(range(10)),
        "y": list(range(0, 100, 10))
    })
    filtered = df[df["x"] > 5]
    print("\nFiltered DataFrame (x > 5):")
    print(filtered)


def read_covid_csv(filepath):
    """Read the COVID-19 CSV file and return a cuDF DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        cuDF DataFrame with all columns from the CSV.
    """
    return cudf.read_csv(filepath)


def process_covid_data(df):
    """Aggregate COVID-19 data by continent.

    Drops rows without a continent, fills missing numeric values with 0,
    then sums new_cases and new_deaths per continent, sorted by new_cases
    in descending order.

    Args:
        df: cuDF DataFrame as returned by read_covid_csv.

    Returns:
        cuDF DataFrame with columns continent, new_cases, new_deaths.
    """
    df = df.dropna(subset=["continent"])
    df["new_cases"] = df["new_cases"].fillna(0)
    df["new_deaths"] = df["new_deaths"].fillna(0)

    aggregated = (
        df.groupby("continent")[["new_cases", "new_deaths"]]
        .sum()
        .reset_index()
        .sort_values("new_cases", ascending=False)
    )
    return aggregated


def try_read_csv():
    """Read and process the COVID-19 CSV file, then print a summary."""
    filepath = os.path.join(os.path.dirname(__file__), "data", "covid_data.csv")
    df = read_covid_csv(filepath)
    print(f"\nCOVID-19 CSV loaded: {len(df):,} rows, {len(df.columns)} columns")

    summary = process_covid_data(df)
    print("\nTotal new cases and deaths by continent:")
    print(summary.to_pandas().to_string(index=False))


if __name__ == "__main__":
    # Print device information and cuDF version
    print(f"cuDF version: {cudf.__version__}")
    device = cp.cuda.Device(0)
    # CuPy exposes CUDA runtime attributes via a C extension that pylint cannot introspect.
    props = cp.cuda.runtime.getDeviceProperties(device.id)  # pylint: disable=c-extension-no-member
    print(f"GPU device: {props['name'].decode()}")
    print(f"Total memory: {props['totalGlobalMem'] / 1024 ** 3:.1f} GB")
    print(f"CUDA capability: {props['major']}.{props['minor']}\n")
    try_series()
    try_dataframe()
    try_numpy_interop()
    try_filter()
    try_read_csv()
    print("\nDemo successful!")
