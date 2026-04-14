"""Pytest configuration – runs before any test module is imported."""
import os

# Use KvikIO compatible mode so cuFile/GDS is never initialised on systems
# that do not have GPUDirect Storage installed.
os.environ.setdefault("KVIKIO_COMPAT_MODE", "ON")

