Configuration Guide
====================

This page explains how to adapt pyOMA-Monitoring to a new monitored
structure.  There are two independent pieces to configure:

1. **``config.yaml``** — static paths and channel lists for the site.
2. **A new ``site_*.py``** — the Python callbacks that the engine calls
   for every file and every window.

Both pieces must be completed before running ``daily.py``.


.. _config-yaml:

Part 1 — Editing config.yaml
------------------------------

``config.yaml`` holds everything that does not require Python logic: file
system roots, quantity-to-origin mappings, channel definitions, plausibility
ranges, and the earliest available data date per origin.  ``config.py``
loads and validates this file at import time and re-exports every value as a
module-level attribute.

A minimal annotated ``config.yaml``:

.. code-block:: yaml

   # Absolute paths — change these for your deployment
   paths:
     file_root:      '/data/raw/'           # root for raw measurement files
     slice_root:     '/data/slices/'        # root for cached preprocessed slices
     db_root:        '/data/results/'       # root for NetCDF result databases
     modal_conf_dir: '/data/modal_conf/'    # root for OMA configuration files

   # Quantity → origin mapping.
   # The "origin tag" is used as a key in subpaths, dtstarts, and channels.
   origins:
     accel: accel
     wind:  wind
     temp:  temp

   # Origin tag → relative subdirectory under paths.file_root
   subpaths:
     accel: 'accel/'
     wind:  'wind/'
     temp:  'temp/'

   # Earliest available data per origin (ISO date string)
   dtstarts:
     accel: '2020-01-01'
     wind:  '2020-01-01'
     temp:  '2020-01-01'

   # Channel definitions
   channels:
     # Required channels per quantity — slices missing any of these are discarded
     all:
       accel: ['Accel_01', 'Accel_02', 'Accel_03', 'Accel_04']
       wind:  ['Wg', 'Wr']
       temp:  ['Pt100_01']
     # Optional channels — may be present but are not required
     optional:
       accel: []
       wind:  []
       temp:  []
     # Lists used by strain callbacks (leave empty if not applicable)
     strain_list: []
     temp_list:   []

   # Physical plausibility ranges [min, max].
   # Slices with any channel outside its range are flagged as erroneous.
   ranges:
     Accel_01: [-20.0, 20.0]
     Accel_02: [-20.0, 20.0]
     Wg:       [0.0,   60.0]

   # FBG-specific fields — leave as empty dicts if not using FBG sensors
   strain_temperature_map: {}
   initial_wavelengths: {}

``config.py`` validates the structure on load.  Missing required keys or
wrong types raise a descriptive ``ValueError`` before any processing starts.


.. _site-module-guide:

Part 2 — Writing a site_*.py
------------------------------

``site_example.py`` is a fully documented template.  Copy it, rename it,
and fill in the stubs.

.. code-block:: bash

   cp site_example.py site_mysite.py

The minimal set of things to implement before running the pipeline:

1. **Configuration constants** — database paths, origins, channel lists
   (can mirror what is in ``config.yaml`` or be hardcoded in the site module).
2. **``file_list_fn``** — discovers raw files on disk for a given origin.
3. **``sync_policy``** — corrects raw timestamps (if the embedded clock is
   unreliable, return ``file_time - duration``; otherwise return
   ``start_time`` unchanged).
4. **``register_<mysite>_site()``** — builds the ``Site`` dataclass and calls
   ``register_site`` / ``set_active_site``.

Transforms (``transforms``), OMA channel mapping (``setup_prep``),
kurtosis thresholds (``error_rules``), and custom channel means
(``channel_mean_fn``) are all optional and can be added later as needed.


Minimal worked template
~~~~~~~~~~~~~~~~~~~~~~~~

The following is a self-contained, runnable site module for a hypothetical
building with two accelerometers and one temperature sensor.  It is
distilled from ``site_example.py``:

.. code-block:: python

   """site_building.py — minimal site module for a two-accelerometer building."""
   import glob
   import os
   import logging

   import numpy as np

   logger = logging.getLogger(__name__)

   # -- Configuration ---------------------------------------------------------

   DB_ROOT    = '/data/building/results/'
   SLICE_ROOT = '/data/building/slices/'
   CONF_DIR   = '/data/building/modal_conf/'
   FILE_ROOT  = '/data/building/raw/'

   ORIGINS  = {'accel': 'accel', 'temp': 'temp'}
   SUBPATHS = {'accel': 'accel/', 'temp': 'temp/'}

   ALL_CHANNELS      = {'accel': ['Accel_01', 'Accel_02'], 'temp': ['Pt100_01']}
   OPTIONAL_CHANNELS = {'accel': [], 'temp': []}
   DTSTARTS          = {'accel': '2024-01-01', 'temp': '2024-01-01'}
   RANGES            = {'Accel_01': (-20.0, 20.0), 'Accel_02': (-20.0, 20.0)}
   ERROR_RULES       = {'accel': {'kurtosis_max': 5, 'kurtosis_min': -2}}
   MODAL_BANDS       = [(1.0, 1.5), (2.5, 3.0)]

   # -- File discovery --------------------------------------------------------

   def _building_get_file_list(origin, reduced=False, file_info=None):
       path = os.path.join(FILE_ROOT, SUBPATHS[origin])
       file_list = glob.glob(os.path.join(path, '*.dat'))
       if reduced and file_info is not None and 'file_name' in file_info:
           existing = set(file_info['file_name'].values.astype(str))
           file_list = [f for f in file_list
                        if os.path.basename(f) not in existing]
       return file_list

   # -- Sync policy -----------------------------------------------------------

   def _building_sync_policy(start_time, file_time, duration):
       return start_time    # embedded clock is reliable

   # -- OMA channel mapping ---------------------------------------------------

   def _building_setup_accel(headers):
       ref_channels  = [headers.index('Accel_01')]
       accel_channels = list(range(len(headers)))
       disp_channels  = []
       chan_dofs_dict = {
           'Accel_01': [1, 0,   0],
           'Accel_02': [2, 90,  0],
       }
       return ref_channels, accel_channels, disp_channels, chan_dofs_dict

   # -- Registration ----------------------------------------------------------

   def register_building_site():
       import monitoring as _m

       site = _m.Site(
           name='building',
           transforms={},
           setup_prep={'accel': _building_setup_accel},
           error_rules=ERROR_RULES,
           sync_policy=_building_sync_policy,
           modal_bands=MODAL_BANDS,
           file_list_fn=_building_get_file_list,
           channel_mean_fn=None,
           preproc_channels={'accel': ['Accel_01', 'Accel_02']},
           db_root_path=DB_ROOT,
           slice_root_path=SLICE_ROOT,
           modal_conf_dir=CONF_DIR,
           file_root_path=FILE_ROOT,
           origins=ORIGINS,
           subpaths=SUBPATHS,
           all_channels=ALL_CHANNELS,
           optional_channels=OPTIONAL_CHANNELS,
           dtstarts=DTSTARTS,
           ranges=RANGES,
       )
       _m.register_site(site)
       _m.set_active_site(site)

   register_building_site()

Then wire it into ``daily.py`` by replacing the existing site import::

    import site_building  # registers and activates on import

And run:

.. code-block:: bash

   python daily.py -d 120 -q accel --file_info --stats --modal --plot


OMA configuration files
~~~~~~~~~~~~~~~~~~~~~~~~

``modal_conf_dir/<quantity>/`` must contain the pyOMA geometry and SSI
configuration files.  See the
`pyOMA input-file format documentation
<https://py-oma.readthedocs.io/en/latest/input_file_formats.html>`_ for the
exact format of each file.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``nodes``
     - Node coordinates used by ``GeometryProcessor``.
   * - ``lines``
     - Line connectivity between nodes.
   * - ``master_slaves``
     - Master/slave channel relationships (may be empty).
   * - ``ssi_config``
     - SSI parameters: model-order range, number of block rows, etc.
   * - ``channel_dofs``
     - Channel-to-DOF mapping (auto-generated from ``setup_prep`` output on
       first run; do not edit by hand).


Graceful failure when no site is registered
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you import ``monitoring`` without first importing a site module,
``_active_site`` is ``None``.  Engine functions that can proceed without a
site (statistics, file reading) do so with safe defaults.  Functions that
require a site-specific hook raise a ``RuntimeError`` with a descriptive
message::

    RuntimeError: No file_list_fn registered.
    Import the appropriate site module first.

This makes the failure mode explicit: the error message names the missing
step, rather than producing silent incorrect results.
