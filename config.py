"""YAML-based configuration loader for the tower monitoring pipeline.

All static site configuration lives in ``config.yaml`` next to this file.
This module loads that file, validates its structure, and re-exports every
value as a module-level attribute so that existing code using
``import config; config.origins`` continues to work without modification.

Runtime-only state (``file_cache``, ``ds_cache``, ``pid``) is also
initialised here and is *not* part of the YAML file.
"""
import datetime
import logging
import os
from collections import deque

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str = _CONFIG_PATH) -> dict:
    """Load *path* as YAML and return the validated configuration dict.

    Parameters
    ----------
    path:
        Path to a YAML file with the structure defined in ``config.yaml``.

    Raises
    ------
    FileNotFoundError
        When *path* does not exist.
    yaml.YAMLError
        When the file is not valid YAML.
    ValueError
        When required keys are missing or values have wrong types.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path!r}")
    with open(path, 'r', encoding='utf-8') as fh:
        raw = yaml.safe_load(fh)
    validate_config(raw)
    return raw


def validate_config(cfg: dict) -> None:
    """Validate the structure of a loaded configuration dict.

    Raises :exc:`ValueError` with a descriptive message on the first
    violation found.

    Parameters
    ----------
    cfg:
        Dict produced by :func:`load_config` (or ``yaml.safe_load``).
    """
    _require_keys(cfg, ['paths', 'origins', 'subpaths', 'dtstarts',
                        'channels', 'ranges', 'strain_temperature_map',
                        'initial_wavelengths'])
    _validate_paths(cfg['paths'])
    _validate_string_map(cfg['origins'], 'origins')
    _validate_string_map(cfg['subpaths'], 'subpaths')
    _validate_dtstarts(cfg['dtstarts'])
    _validate_channels(cfg['channels'])
    _validate_ranges(cfg['ranges'])
    _validate_strain_temperature_map(cfg['strain_temperature_map'])
    _validate_initial_wavelengths(cfg['initial_wavelengths'])


# ---------------------------------------------------------------------------
# Private helpers — validators
# ---------------------------------------------------------------------------

def _require_keys(mapping: dict, keys: list) -> None:
    """Raise ``ValueError`` if any key in *keys* is absent from *mapping*.

    Parameters
    ----------
    mapping : dict
        The configuration dict (or sub-dict) to inspect.
    keys : list of str
        Required keys that must be present in *mapping*.

    Raises
    ------
    ValueError
        Lists every missing key in the error message.
    """
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")


def _validate_paths(paths: dict) -> None:
    """Validate the ``paths`` section of the configuration.

    Parameters
    ----------
    paths : dict
        Must contain the keys ``file_root``, ``slice_root``, ``db_root``,
        and ``modal_conf_dir``, each mapping to a string.

    Raises
    ------
    ValueError
        When a required key is absent or a value is not a string.
    """
    _require_keys(paths, ['file_root', 'slice_root', 'db_root', 'modal_conf_dir'])
    for key, val in paths.items():
        if not isinstance(val, str):
            raise ValueError(
                f"paths.{key} must be a string, got {type(val).__name__!r}")


def _validate_string_map(mapping: dict, section: str) -> None:
    """Validate that every value in *mapping* is a string.

    Parameters
    ----------
    mapping : dict
        A flat dict whose values must all be strings.
    section : str
        Section name used in error messages (e.g. ``'origins'``).

    Raises
    ------
    ValueError
        When any value is not a string.
    """
    for key, val in mapping.items():
        if not isinstance(val, str):
            raise ValueError(
                f"{section}.{key} must be a string, got {type(val).__name__!r}")


def _validate_dtstarts(dtstarts: dict) -> None:
    """Validate the ``dtstarts`` section (earliest available data per origin).

    Parameters
    ----------
    dtstarts : dict
        Maps origin tag to an ISO date string or ``datetime.date`` object.

    Raises
    ------
    ValueError
        When a value is not a string/date or cannot be parsed as an ISO date.
    """
    for key, val in dtstarts.items():
        if not isinstance(val, (str, datetime.date)):
            raise ValueError(
                f"dtstarts.{key} must be an ISO date string, "
                f"got {type(val).__name__!r}")
        try:
            datetime.date.fromisoformat(str(val))
        except ValueError as exc:
            raise ValueError(
                f"dtstarts.{key} is not a valid ISO date: {val!r}") from exc


def _validate_channels(channels: dict) -> None:
    """Validate the ``channels`` section (required/optional channel lists).

    Parameters
    ----------
    channels : dict
        Must contain the keys ``all``, ``optional``, ``strain_list``, and
        ``temp_list``.  Values under ``all`` must be lists of strings.

    Raises
    ------
    ValueError
        When a required key is absent, a channel group is not a list, or a
        channel name is not a string.
    """
    _require_keys(channels, ['all', 'optional', 'strain_list', 'temp_list'])
    for grp, chans in channels['all'].items():
        if not isinstance(chans, list):
            raise ValueError(f"channels.all.{grp} must be a list")
        for ch in chans:
            if not isinstance(ch, str):
                raise ValueError(
                    f"channels.all.{grp}: every entry must be a string, "
                    f"got {ch!r}")
    for lst_name in ('strain_list', 'temp_list'):
        if not isinstance(channels[lst_name], list):
            raise ValueError(f"channels.{lst_name} must be a list")


def _validate_ranges(ranges: dict) -> None:
    """Validate the ``ranges`` section (physical plausibility bounds per channel).

    Parameters
    ----------
    ranges : dict
        Maps channel name to a ``[min, max]`` two-element list or tuple of
        numbers.

    Raises
    ------
    ValueError
        When a value is not a two-element list, contains non-numbers, or has
        ``min > max``.
    """
    for key, val in ranges.items():
        if not (isinstance(val, (list, tuple)) and len(val) == 2):
            raise ValueError(
                f"ranges.{key} must be a [min, max] pair, got {val!r}")
        if not all(isinstance(v, (int, float)) for v in val):
            raise ValueError(
                f"ranges.{key} bounds must be numbers, got {val!r}")
        if val[0] > val[1]:
            raise ValueError(
                f"ranges.{key}: min ({val[0]}) must not exceed max ({val[1]})")


def _validate_strain_temperature_map(stmap: dict) -> None:
    """Validate the ``strain_temperature_map`` section.

    Each entry maps a strain-channel group to a dict of
    ``{strain_channel: temperature_channel}`` string pairs, used by FBG
    correction routines to look up which temperature channel compensates a
    given strain channel.

    Parameters
    ----------
    stmap : dict
        The ``strain_temperature_map`` sub-dict from the YAML config.  May be
        an empty dict if FBG sensors are not used.

    Raises
    ------
    ValueError
        When a group value is not a dict or a temperature-channel value is
        not a string.
    """
    for grp, mapping in stmap.items():
        if not isinstance(mapping, dict):
            raise ValueError(
                f"strain_temperature_map.{grp} must be a dict, "
                f"got {type(mapping).__name__!r}")
        for strain_ch, temp_ch in mapping.items():
            if not isinstance(temp_ch, str):
                raise ValueError(
                    f"strain_temperature_map.{grp}.{strain_ch} must be a "
                    f"string, got {type(temp_ch).__name__!r}")


def _validate_initial_wavelengths(wl: dict) -> None:
    """Validate the ``initial_wavelengths`` section (FBG reference wavelengths).

    Parameters
    ----------
    wl : dict
        Maps FBG channel name to its reference wavelength in nanometres.
        May be an empty dict if FBG sensors are not used.

    Raises
    ------
    ValueError
        When any wavelength value is not a number.
    """
    for key, val in wl.items():
        if not isinstance(val, (int, float)):
            raise ValueError(
                f"initial_wavelengths.{key} must be a number, "
                f"got {type(val).__name__!r}")


# ---------------------------------------------------------------------------
# Private helpers — converters
# ---------------------------------------------------------------------------

def _parse_dtstarts(raw: dict) -> dict:
    """Convert ISO date strings (or ``datetime.date`` objects) to ``datetime.datetime``."""
    result = {}
    for key, val in raw.items():
        result[key] = datetime.datetime.fromisoformat(str(val))
    return result


def _parse_ranges(raw: dict) -> dict:
    """Convert ``[min, max]`` lists to ``(min, max)`` tuples."""
    return {k: tuple(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Load and expose configuration
# ---------------------------------------------------------------------------
_cfg = load_config()

# Paths
file_root_path: str = _cfg['paths']['file_root']
slice_root_path: str = _cfg['paths']['slice_root']
db_root_path: str = _cfg['paths']['db_root']
modal_conf_dir: str = _cfg['paths']['modal_conf_dir']

# Quantity → origin / sub-directory mappings
origins: dict = _cfg['origins']
subpaths: dict = _cfg['subpaths']

# Earliest available data per origin
dtstarts: dict = _parse_dtstarts(_cfg['dtstarts'])

# Channel definitions
all_channels: dict = _cfg['channels']['all']
optional_channels: dict = _cfg['channels']['optional']
strain_channels: list = _cfg['channels']['strain_list']
temp_channels: list = _cfg['channels']['temp_list']

# Physical plausibility ranges as (min, max) tuples
ranges: dict = _parse_ranges(_cfg['ranges'])

# Strain ↔ temperature channel mapping
strain_t: dict = _cfg['strain_temperature_map']

# Initial FBG wavelengths [nm]
initial_wl: dict = _cfg['initial_wavelengths']

# ---------------------------------------------------------------------------
# Runtime-only state (not part of YAML configuration)
# ---------------------------------------------------------------------------
pid: str = str(os.getpid())
file_cache: deque = deque(maxlen=25)

ds_cache: dict = {}
for _origin in set(origins.values()):
    ds_cache[f'{_origin}_file_info'] = {'ds': None, 'mtime': None}
for _quantity in origins:
    ds_cache[f'{_quantity}_stats'] = {'ds': None, 'mtime': None}
    if _quantity in ('accel', 'strain_rosettes'):
        ds_cache[f'{_quantity}_modal'] = {'ds': None, 'mtime': None}
