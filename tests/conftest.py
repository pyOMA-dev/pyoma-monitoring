"""Shared pytest fixtures for the pyOMA-Monitoring test suite.

Adds the package directory to sys.path so the non-package modules
(monitoring.py, config.py, etc.) can be imported, and provides
fixtures that redirect the pipeline to the short result_db so tests
do not touch production databases.
"""
import os
import sys

import numpy as np
import pytest

# Allow ``import monitoring``, ``import config``, etc.
PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

# Register the Geyer site so engine functions that look up the active site
# behave correctly in the test suite (all test data is Geyer data).
import site_geyer  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Paths to test data (read-only)
# Falls back to the developer's external data directory if the in-repo
# copy does not exist (e.g. before the first git clone on a new machine).
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TOWERDATA_DIR = os.path.join(_DATA_DIR, "towerdata")
RESULT_DB_SHORT = os.path.join(_DATA_DIR, "result_db_short")
FILE_INFO_DIR = RESULT_DB_SHORT  # file_info_*.nc live here


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require on-disk data (may be slow)",
    )


def data_available():
    return os.path.isdir(TOWERDATA_DIR) and os.path.isdir(RESULT_DB_SHORT)


skip_if_no_data = pytest.mark.skipif(
    not data_available(),
    reason="on-disk test data not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def short_db(monkeypatch):
    """Redirect config.db_root_path to the short (read-only) result database.

    Also clears the ds_cache so stale cached datasets from the production
    db_root_path cannot bleed into integration tests.
    """
    import config

    monkeypatch.setattr(config, "db_root_path", RESULT_DB_SHORT)

    # Invalidate any cached datasets (mtime-keyed dicts in config.ds_cache)
    for key in config.ds_cache:
        config.ds_cache[key]["ds"] = None
        config.ds_cache[key]["mtime"] = None

    yield config

    # monkeypatch restores db_root_path automatically on teardown


@pytest.fixture()
def first_accel_file():
    """Return the path to the earliest accelerometer .dat file."""
    path = os.path.join(
        TOWERDATA_DIR,
        "Accel_continuously__0_2026-06-01_00-00-00_000000.dat",
    )
    if not os.path.exists(path):
        pytest.skip(f"Test file not found: {path}")
    return path


@pytest.fixture()
def synthetic_measurement():
    """Return a small reproducible measurement array with four channels."""
    rng = np.random.default_rng(0)
    meas = rng.standard_normal((500, 4))
    headers = ["Ch_A", "Ch_B", "Ch_C", "Ch_D"]
    return meas, headers
