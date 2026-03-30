"""Demo script to test cuDF on GPU."""
import cudf
import cupy as cp
import numpy as np


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
    print("\nAll tests passed!")
