"""Example / template site module for pyOMA-Monitoring.

Copy this file, rename it ``site_<yoursite>.py``, and fill in every
``raise NotImplementedError`` stub with your site-specific logic.  Import the
finished module once (e.g. at the top of ``daily.py``) to register and activate
your site in the monitoring engine::

    import site_mysite  # registers on import

All engine functions in ``monitoring.py`` will then use your callbacks
automatically.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Site configuration — replace every placeholder string / dict with real values
# ---------------------------------------------------------------------------

#: Absolute path to the root directory that holds the result databases
#: (``file_info_*.nc``, ``stats_*.nc``, ``modal_*.nc``).
EXAMPLE_DB_ROOT_PATH: str = "/path/to/results/db/"

#: Absolute path to the root directory used for pre-processed signal slices.
EXAMPLE_SLICE_ROOT_PATH: str = "/path/to/slices/"

#: Absolute path to the directory that contains the OMA configuration files
#: (``nodes``, ``lines``, ``master_slaves``, ``ssi_config``, …).
EXAMPLE_MODAL_CONF_DIR: str = "/path/to/modal_conf/"

#: Absolute path to the root directory that holds the raw measurement files.
EXAMPLE_FILE_ROOT_PATH: str = "/path/to/raw_data/"

#: Map *quantity* → *origin tag*.  The origin tag is the key used in
#: :attr:`subpaths`, :attr:`all_channels`, and :attr:`dtstarts`.
#: Add one entry per measured quantity.
#:
#: Example::
#:
#:     EXAMPLE_ORIGINS = {
#:         "accel": "accel",
#:         "wind":  "wind",
#:     }
EXAMPLE_ORIGINS: Dict[str, str] = {
    "accel": "accel",
    # add more quantities here
}

#: Map *origin tag* → relative sub-path under :data:`EXAMPLE_FILE_ROOT_PATH`.
EXAMPLE_SUBPATHS: Dict[str, str] = {
    "accel": "accel/",
    # add more origins here
}

#: Map *quantity* → list of channel names that *must* be present in every slice.
EXAMPLE_ALL_CHANNELS: Dict[str, List[str]] = {
    "accel": ["Accel_01", "Accel_02"],
    # add more quantities here
}

#: Map *quantity* → list of channel names that *may* be present (not required).
EXAMPLE_OPTIONAL_CHANNELS: Dict[str, List[str]] = {
    "accel": [],
    # add more quantities here
}

#: Map *origin tag* → earliest available datetime (ISO string or
#: ``datetime.datetime``).  Slices before this timestamp are skipped.
EXAMPLE_DTSTARTS: Dict[str, object] = {
    "accel": "2020-01-01",
    # add more origins here
}

#: Map *channel name* → ``(min, max)`` plausibility range.  Samples outside
#: this interval cause the affected time slice to be flagged as erroneous.
EXAMPLE_RANGES: Dict[str, Tuple[float, float]] = {
    "Accel_01": (-20.0, 20.0),
    # add more channels here
}

# ---------------------------------------------------------------------------
# Error rules
# ---------------------------------------------------------------------------

#: Map *quantity* → dict of kurtosis thresholds used by ``describe_stats``.
#: Slices whose kurtosis falls outside ``[kurtosis_min, kurtosis_max]`` are
#: flagged as erroneous.  Omit a quantity to apply no kurtosis check.
EXAMPLE_ERROR_RULES: Dict[str, dict] = {
    "accel": {"kurtosis_max": 5, "kurtosis_min": -2},
}

# ---------------------------------------------------------------------------
# Modal frequency bands
# ---------------------------------------------------------------------------

#: List of ``(f_lo, f_hi)`` Hz bands passed to ``split_modepairs`` when
#: assigning poles to named modes.  Add one tuple per expected mode.
EXAMPLE_MODAL_BANDS: List[Tuple[float, float]] = [
    (0.5, 1.0),
    # add more bands here
]

# ---------------------------------------------------------------------------
# Pre-processing channel selection
# ---------------------------------------------------------------------------

#: Map *quantity* → list of channel names kept after the band-pass /
#: decimation pre-processing step.  Only relevant for quantities that go
#: through OMA (typically ``"accel"`` and strain-type quantities).
EXAMPLE_PREPROC_CHANNELS: Dict[str, List[str]] = {
    "accel": ["Accel_01", "Accel_02"],
    # add more quantities here
}

# ---------------------------------------------------------------------------
# Transform callbacks
# ---------------------------------------------------------------------------

def _example_accel_transform(
    start_time,
    headers: List[str],
    units: List[str],
    end_time,
    sample_rate: float,
    measurement: np.ndarray,
    quantity: Optional[str] = None,
    **kwargs,
):
    """Apply site-specific post-processing to a raw acceleration slice.

    The function receives a *Slice* — a 6-tuple produced by
    ``monitoring.get_slice`` — and must return a (possibly modified) *Slice*
    of the same shape, or ``None`` to discard the slice entirely.

    Parameters
    ----------
    start_time:
        Timezone-aware ``datetime`` of the first sample.
    headers:
        List of channel names, length ``N``.
    units:
        List of unit strings, length ``N``.
    end_time:
        Timezone-aware ``datetime`` of the last sample.
    sample_rate:
        Samples per second.
    measurement:
        2-D NumPy array of shape ``(T, N)``.
    quantity:
        The quantity key (e.g. ``"accel"``), injected by the engine.
    **kwargs:
        Additional keyword arguments passed through from the engine (e.g.
        ``start_time_local``, ``duration``, ``file_info_temp``).

    Returns
    -------
    tuple | None
        Modified ``(start_time, headers, units, end_time, sample_rate,
        measurement)`` 6-tuple, or ``None`` to drop the slice.
    """
    raise NotImplementedError("Implement site-specific acceleration transform here")


# ---------------------------------------------------------------------------
# setup_prep callbacks
# ---------------------------------------------------------------------------

def _example_setup_accel(headers: List[str]):
    """Return the OMA channel-role mapping for acceleration data.

    Called once per modal-analysis slice, immediately before constructing a
    ``PreProcessSignals`` object.

    Parameters
    ----------
    headers:
        Ordered list of channel names present in the slice.

    Returns
    -------
    ref_channels : list[int]
        Indices into *headers* of the reference (roving-reference) channels.
    accel_channels : list[int]
        Indices of all acceleration channels.
    disp_channels : list[int]
        Indices of all displacement channels (empty if none).
    chan_dofs_dict : dict[str, list]
        Mapping *channel name* → ``[node_id, azimuth_deg, inclination_deg]``
        used to assemble the ``channel_dofs`` file consumed by pyOMA.
    """
    raise NotImplementedError("Implement site-specific OMA channel mapping here")


# ---------------------------------------------------------------------------
# Synchronisation-era policy
# ---------------------------------------------------------------------------

def _example_sync_policy(start_time, file_time, duration):
    """Return the synchronised start time for this recording.

    The monitoring engine calls this for every file it reads.  The default
    implementation simply returns *start_time* unchanged.  Override this
    function when the DAQ clock differs from wall time in a known, era-dependent
    way (e.g. the file's filesystem timestamp is more reliable than the embedded
    timestamp during a specific period).

    Parameters
    ----------
    start_time:
        Timezone-aware start time embedded in the file header.
    file_time:
        Timezone-aware filesystem modification time of the file.
    duration:
        ``datetime.timedelta`` length of the recording.

    Returns
    -------
    datetime.datetime
        The corrected, timezone-aware start time to use as the time coordinate
        in all databases.
    """
    return start_time  # default: trust the embedded timestamp


# ---------------------------------------------------------------------------
# File-list discovery
# ---------------------------------------------------------------------------

def _example_get_file_list(
    origin: str,
    reduced: bool = False,
    file_info=None,
) -> List[str]:
    """Return absolute paths of all data files for *origin*.

    The engine calls this callback with ``reduced=True`` during incremental
    updates (only new files) and with ``reduced=False`` for a full scan.

    Parameters
    ----------
    origin:
        Origin tag (a value from :data:`EXAMPLE_ORIGINS`).
    reduced:
        When ``True`` and *file_info* is provided, return only files not yet
        present in *file_info*.
    file_info:
        ``xarray.Dataset`` of already-processed files (the current
        ``file_info_<origin>.nc`` database).  May be ``None``.

    Returns
    -------
    list[str]
        Absolute file paths to process.
    """
    raise NotImplementedError("Implement site-specific file discovery here")


# ---------------------------------------------------------------------------
# Circular-mean helper (optional)
# ---------------------------------------------------------------------------

def _example_channel_mean_fn(
    header: str,
    measurement: np.ndarray,
    headers: List[str],
) -> Optional[float]:
    """Return a custom mean for *header*, or ``None`` to use the arithmetic mean.

    Used by ``describe_stats`` when computing summary statistics.  Implement
    this for directional channels (e.g. wind direction) where the arithmetic
    mean is undefined.

    Parameters
    ----------
    header:
        Name of the channel whose mean is being computed.
    measurement:
        Full 2-D measurement array of shape ``(T, N)``.
    headers:
        List of all channel names, length ``N``.

    Returns
    -------
    float | None
        The custom mean value, or ``None`` to fall back to the arithmetic mean.
    """
    return None  # default: use arithmetic mean for all channels


# ---------------------------------------------------------------------------
# Site registration
# ---------------------------------------------------------------------------

def register_example_site() -> None:
    """Register and activate the example site in the monitoring engine.

    Called automatically when this module is imported.  Safe to call multiple
    times — it simply overwrites the registry entry for ``"example"``.

    Raises
    ------
    NotImplementedError
        Any stub callback that has not been implemented yet will raise this
        error at runtime when the engine first invokes it.
    """
    import monitoring as _m  # late import avoids circular dependency at load time

    example_site = _m.Site(
        name="example",
        # --- processing callbacks ---
        transforms={
            "accel": _example_accel_transform,
            # add more quantities here; omit a quantity for no-op processing
        },
        setup_prep={
            "accel": _example_setup_accel,
            # add more quantities here
        },
        error_rules=EXAMPLE_ERROR_RULES,
        sync_policy=_example_sync_policy,
        modal_bands=EXAMPLE_MODAL_BANDS,
        file_list_fn=_example_get_file_list,
        channel_mean_fn=_example_channel_mean_fn,
        preproc_channels=EXAMPLE_PREPROC_CHANNELS,
        # --- site configuration ---
        db_root_path=EXAMPLE_DB_ROOT_PATH,
        slice_root_path=EXAMPLE_SLICE_ROOT_PATH,
        modal_conf_dir=EXAMPLE_MODAL_CONF_DIR,
        file_root_path=EXAMPLE_FILE_ROOT_PATH,
        origins=EXAMPLE_ORIGINS,
        subpaths=EXAMPLE_SUBPATHS,
        all_channels=EXAMPLE_ALL_CHANNELS,
        optional_channels=EXAMPLE_OPTIONAL_CHANNELS,
        dtstarts=EXAMPLE_DTSTARTS,
        ranges=EXAMPLE_RANGES,
    )

    _m.register_site(example_site)
    _m.set_active_site(example_site)


register_example_site()
