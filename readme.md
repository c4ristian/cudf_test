# cudf_test

A minimal project to test [RAPIDS cuDF](https://docs.rapids.ai/api/cudf/stable/) on GPU.

## Setup

```bash
conda env create -f environment.yml
conda activate cudf_test
```

## Run

```bash
python cudf_demo.py
pytest tests/
```

## Lint

```bash
pylint cudf_demo.py tests/test_cudf.py
```

You can also lint all Python files from the project root:

```bash
pylint *.py tests/*.py
```

## Requirements

- NVIDIA GPU with CUDA 12.x driver support
- Driver version 520+ recommended

