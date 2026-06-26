.. _site-module:

Writing a Site Module
=====================

A *site module* is the single file that encapsulates everything specific to one
monitoring deployment.  The generic engine in ``monitoring.py`` is completely
site-agnostic; it delegates every site-dependent decision to callbacks
registered through the :class:`~monitoring.Site` dataclass.

``site_example.py`` is a fully-documented template.  Copy it, rename it
``site_<yoursite>.py``, fill in the stubs, and you have a working site.

.. code-block:: python

   import site_mysite          # registers and activates the site on import
   import monitoring

   site = monitoring.get_active_site()
   print(site.name)            # "mysite"


Architecture overview
---------------------

.. code-block:: text

   daily.py / notebook
       │  import site_mysite
       ▼
   site_mysite.py
       │  reads paths / channel lists from a config file (or hard-codes them)
       │  defines transform / setup_prep / sync_policy / file_list_fn callbacks
       │  calls register_site() + set_active_site()  ──►  monitoring.py
       ▼
   monitoring.py   (site-agnostic engine)
       │  get_file_list()          uses  site.file_list_fn
       │  get_slice_corrected()    uses  site.transforms
       │  describe_stats()         uses  site.ranges, site.error_rules,
       │                                 site.channel_mean_fn
       │  create_stats()           uses  site.origins, site.dtstarts
       │  modal_analysis_single()  uses  site.setup_prep
       └──────────────────────────────────────────────────────────────────────

The dependency arrow runs in one direction only: the site module imports the
engine (lazily, inside functions), never the other way around.


The ``Site`` dataclass
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Field
     - Type
     - Purpose
   * - ``name``
     - ``str``
     - Unique site identifier; used as registry key.
   * - ``transforms``
     - ``Dict[str, Callable]``
     - Per-quantity post-processing applied to every raw slice before it is
       stored.  See :ref:`transform-callback`.
   * - ``setup_prep``
     - ``Dict[str, Callable]``
     - Per-quantity channel-role mapping fed into ``PreProcessSignals`` before
       OMA.  See :ref:`setup-prep-callback`.
   * - ``error_rules``
     - ``Dict[str, dict]``
     - Kurtosis thresholds per quantity.  Keys: ``kurtosis_max``,
       ``kurtosis_min``.
   * - ``sync_policy``
     - ``Callable``
     - Converts raw file timestamps to a canonical wall-clock time.
       See :ref:`sync-policy-callback`.
   * - ``modal_bands``
     - ``List[tuple]``
     - ``(f_lo, f_hi)`` Hz bands used by ``split_modepairs`` to assign
       poles to named modes.
   * - ``file_list_fn``
     - ``Callable``
     - Discovers raw data files on disk.  See :ref:`file-list-callback`.
   * - ``channel_mean_fn``
     - ``Optional[Callable]``
     - Returns a custom mean for directional channels (e.g. wind direction),
       or ``None`` to fall back to the arithmetic mean.
       See :ref:`channel-mean-callback`.
   * - ``preproc_channels``
     - ``Dict[str, list]``
     - Channel names kept after band-pass / decimation pre-processing per
       quantity.
   * - ``db_root_path``
     - ``str``
     - Root directory for result databases (``file_info_*.nc``, etc.).
   * - ``slice_root_path``
     - ``str``
     - Root directory for pre-processed signal slices (``.npz`` files).
   * - ``modal_conf_dir``
     - ``str``
     - Directory containing OMA configuration files (``nodes``, ``lines``,
       ``ssi_config``, …).
   * - ``file_root_path``
     - ``str``
     - Root directory of raw measurement files.
   * - ``origins``
     - ``Dict[str, str]``
     - Maps *quantity* → *origin tag* (the key used in ``subpaths``,
       ``all_channels``, and ``dtstarts``).
   * - ``subpaths``
     - ``Dict[str, str]``
     - Maps *origin tag* → relative sub-directory under ``file_root_path``.
   * - ``all_channels``
     - ``Dict[str, list]``
     - Required channel names per quantity.  A slice missing any of these is
       discarded.
   * - ``optional_channels``
     - ``Dict[str, list]``
     - Channel names that *may* be present but are not required.
   * - ``dtstarts``
     - ``Dict[str, object]``
     - Earliest available datetime per origin tag.  ISO string or
       ``datetime.datetime``.
   * - ``ranges``
     - ``Dict[str, tuple]``
     - ``(min, max)`` plausibility range per channel name.  Samples outside
       this interval flag the slice as erroneous.


Callback reference
------------------

.. _transform-callback:

``transforms[quantity]`` — slice transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Signature**::

    def my_transform(
        start_time,          # timezone-aware datetime
        headers,             # List[str] — channel names, length N
        units,               # List[str] — unit strings, length N
        end_time,            # timezone-aware datetime
        sample_rate,         # float — samples per second
        measurement,         # np.ndarray — shape (T, N)
        quantity=None,       # str — injected by the engine
        **kwargs,            # start_time_local, duration, file_info_temp, …
    ):
        ...
        return start_time, headers, units, end_time, sample_rate, measurement
        # or return None  →  slice is discarded

**When it is called**:
  ``monitoring.get_slice_corrected`` applies this transform to every raw slice
  before saving it to disk.  It is called once per time window per quantity.

**Contract**:

* The returned 6-tuple must have the same structure as the input.
  ``headers`` and ``measurement`` may be modified (e.g. columns added /
  removed, values transformed); ``T`` and ``N`` must remain consistent.
* Return ``None`` to signal that the slice is invalid and should not be stored.
* The callback may call back into the engine (e.g. ``monitoring.get_slice``)
  to fetch auxiliary data.  Use a **lazy import** to avoid a circular import
  at module-load time::

      def my_transform(...):
          import monitoring as _m   # inside the function — safe
          aux = _m.get_slice(...)

**Quantities with no transform**:
  Simply omit the quantity from the ``transforms`` dict.  The engine will use
  the raw slice unchanged.

.. _setup-prep-callback:

``setup_prep[quantity]`` — OMA channel-role mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Signature**::

    def my_setup(headers):          # List[str]
        ...
        return ref_channels, accel_channels, disp_channels, chan_dofs_dict

**Return values**:

* ``ref_channels`` — ``List[int]``: indices of reference channels in *headers*.
* ``accel_channels`` — ``List[int]``: indices of all acceleration channels.
* ``disp_channels`` — ``List[int]``: indices of all displacement channels
  (pass ``[]`` when none).
* ``chan_dofs_dict`` — ``Dict[str, list]``: mapping *channel name* →
  ``[node_id, azimuth_deg, inclination_deg]``.

**When it is called**:
  ``monitoring.modal_analysis_single`` calls this once per slice before
  constructing a ``PreProcessSignals`` object.

.. _sync-policy-callback:

``sync_policy`` — timestamp correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Signature**::

    def my_sync_policy(start_time, file_time, duration):
        ...
        return corrected_start_time    # timezone-aware datetime

**Parameters**:

* ``start_time`` — timezone-aware ``datetime``: timestamp embedded in the
  file header.
* ``file_time`` — timezone-aware ``datetime``: filesystem modification time.
* ``duration`` — ``datetime.timedelta``: recording length.

**When it is called**:
  ``monitoring.create_file_info`` calls this for every file, and
  ``monitoring.get_slice`` calls it for every constituent file of a slice.
  The returned value is used as the ``time`` coordinate in all databases.

**Default behaviour**:
  Return *start_time* unchanged when the embedded timestamp is reliable::

      def my_sync_policy(start_time, file_time, duration):
          return start_time

.. _file-list-callback:

``file_list_fn`` — file discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Signature**::

    def my_get_file_list(origin, reduced=False, file_info=None):
        ...
        return ["/abs/path/to/file1", "/abs/path/to/file2", ...]

**Parameters**:

* ``origin`` — ``str``: origin tag (a value from ``Site.origins``).
* ``reduced`` — ``bool``: when ``True``, return only files not yet present in
  *file_info*.
* ``file_info`` — ``xr.Dataset | None``: current ``file_info_<origin>.nc``
  database.  Present when ``reduced=True``; ``None`` otherwise.

**When it is called**:
  ``monitoring.get_file_list`` and ``monitoring.create_file_info`` call this
  to discover which raw files to process.

.. _channel-mean-callback:

``channel_mean_fn`` — custom channel mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Signature**::

    def my_channel_mean_fn(header, measurement, headers):
        ...
        return float | None

**Parameters**:

* ``header`` — ``str``: name of the channel whose mean is needed.
* ``measurement`` — ``np.ndarray`` shape ``(T, N)``: full measurement array.
* ``headers`` — ``List[str]``: all channel names, length ``N``.

**Return value**:
  A ``float`` mean to use, or ``None`` to fall back to the arithmetic mean.

**When it is called**:
  ``monitoring.describe_stats`` calls this once per channel when building
  statistical summaries.  Implement it for directional quantities (e.g. wind
  direction in degrees) where the arithmetic mean wraps around 0°/360°.

  Set ``channel_mean_fn=None`` in :class:`~monitoring.Site` to skip the hook
  entirely and always use the arithmetic mean.


Step-by-step: adding a new site
--------------------------------

1. **Copy the template**::

       cp site_example.py site_mysite.py

2. **Edit the configuration constants** at the top of the file:
   ``MYSITE_DB_ROOT_PATH``, ``MYSITE_ORIGINS``, ``MYSITE_ALL_CHANNELS``, etc.

3. **Implement the callbacks** — replace every ``raise NotImplementedError``
   with real logic.  Start with ``file_list_fn`` (needed to scan files) and
   ``sync_policy`` (needed to stamp them correctly).  Add ``transforms`` and
   ``setup_prep`` only for quantities that need them.

4. **Register the site** — ``register_example_site()`` is called automatically
   at module-import time.  Rename it to ``register_mysite_site()`` and update
   the ``name`` field.

5. **Wire it into ``daily.py``**::

       import site_mysite  # noqa: F401  — registers on import

6. **Test incrementally**:

   .. code-block:: bash

       python -m pytest tests/ -x -q

   The test suite patches ``monitoring._active_site`` via ``monkeypatch``, so
   your callbacks are isolated from the real file system during unit tests.


Configuration files required for OMA
--------------------------------------

``modal_conf_dir/<quantity>/`` must contain:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``nodes``
     - Node coordinates (pyOMA ``GeometryProcessor`` format).
   * - ``lines``
     - Line connectivity between nodes.
   * - ``master_slaves``
     - Master/slave channel relationships (may be empty).
   * - ``ssi_config``
     - SSI algorithm parameters (model-order range, block rows, …).
   * - ``channel_dofs``
     - Channel-to-DOF mapping (auto-generated from ``setup_prep`` output by
       ``modal_analysis_single``; do not edit by hand).

See the pyOMA documentation for the exact file format of each configuration
file.
