=======================
pyOMA-Monitoring
=======================

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License: MIT

pyOMA-Monitoring is the **application layer** for long-term structural health
monitoring with `pyOMA <https://py-oma.readthedocs.io>`_.  It orchestrates the
daily data pipeline — ingestion, quality assessment, signal preprocessing, and
automated modal analysis — and stores every result in a time-indexed xarray /
NetCDF database.

If you are looking for the OMA algorithms (SSI, pLSCF, ERA, stabilisation
diagrams), those live in the **pyOMA library** at
https://py-oma.readthedocs.io.  This documentation covers only the monitoring
application: how it reads files, manages the database, runs the pipeline, and
how to adapt it to a new monitored structure.

The system has been running continuously since 2015 on a 190 m
telecommunication tower; see `pyOMA's monitoring page
<https://py-oma.readthedocs.io/en/latest/continuous_monitoring.html>`_ for
an overview and selected long-term results.


.. ── About ──────────────────────────────────────────────────────────────────


---------------------------
About pyOMA-Monitoring
---------------------------

pyOMA-Monitoring covers the full monitoring workflow from raw binary files on
disk to long-term modal trend charts:

.. list-table::

   * - **File ingestion**
     - Scans raw ``.dat``/``.csv``/``.bin`` files (Gantner Q.Station and
       FBG interrogator formats); reads per-channel statistics into a
       time-indexed file-info database; handles ``.bz2`` compression and an
       in-process LRU file cache.
   * - **Quality assessment**
     - Per-channel plausibility ranges and kurtosis thresholds flag
       erroneous slices before any further processing.
   * - **Signal preprocessing**
     - Fixed-duration windows (10 / 30 / 60 / 120 min) are extracted,
       transformed by site-specific callbacks, bandpass-filtered (0.1–5 Hz),
       and decimated to 10 Hz.  Preprocessed slices are cached as
       ``.npz`` files so interrupted runs can resume.
   * - **Modal analysis**
     - Each valid window is processed by pyOMA's
       :class:`~pyOMA.core.VarSSIRef.VarSSIRef` estimator followed by
       automated stabilisation clustering.
   * - **Result storage**
     - Modal parameters, signal statistics, and environmental quantities are
       merged into sparse xarray / NetCDF databases with three named
       dimensions: **time**, **modes**, and **channels**.
   * - **Multi-worker safety**
     - ``MultiLock`` — a two-layer file-based advisory lock on top of
       ``simpleflock`` — allows several worker processes to write to the
       same NetCDF database concurrently without corruption.


.. ── Install ─────────────────────────────────────────────────────────────────


-------
Install
-------

Requirements: Python ≥ 3.9, plus pyOMA (see
`py-oma.readthedocs.io <https://py-oma.readthedocs.io>`_).

.. code-block:: bash

   git clone https://github.com/pyOMA-dev/pyOMA-Monitoring.git
   cd pyOMA-Monitoring
   pip install -e .

The package installs ``numpy``, ``scipy``, ``matplotlib``, ``pandas``,
``xarray``, ``pytz``, ``tzlocal``, ``python-dateutil``, ``pyyaml``,
``simpleflock``, ``h5netcdf``, ``netcdf4``, and ``pyOMA`` automatically.


.. ── Project structure ────────────────────────────────────────────────────────


-----------------
Project structure
-----------------

::

    pyOMA-Monitoring/
    ├── monitoring.py          # generic engine — site-agnostic pipeline functions
    ├── time_convention.py     # single source of truth for timezone conversions
    ├── config.py              # YAML config loader and validator
    ├── config.yaml            # static site configuration (paths, channels, ranges)
    ├── daily.py               # CLI entry-point: --file_info / --stats / --modal / --plot
    ├── daily2.sh              # cron wrapper: iterates quantities and durations
    ├── site_example.py        # public template for adding a new monitored structure
    ├── site_tower.py          # site-specific callbacks (private; not distributed)
    ├── gantner_reader.py      # Q.Station .dat / .csv reader
    ├── fbg_strain_reader.py   # FBG interrogator .bin / .txt reader
    ├── MultiLock.py           # file-based advisory lock for concurrent NetCDF access
    ├── post_processing.py     # daily / waterfall plot functions
    ├── tests/                 # pytest suite
    └── doc/                   # this documentation


.. ── Toctree ──────────────────────────────────────────────────────────────────


.. toctree::
   :hidden:
   :maxdepth: 2

   architecture
   pipeline
   configuration
   site_module
   api_reference


.. ── API reference ────────────────────────────────────────────────────────────


The full API reference is at :doc:`api_reference`:

* :mod:`monitoring` — engine functions (file ingestion, slicing, statistics, OMA)
* :mod:`time_convention` — :class:`~time_convention.TimeConvention` singleton ``TC``
* :mod:`config` — YAML loader and attribute re-exports
* :mod:`MultiLock` — :class:`~MultiLock.MultiLock` context manager
* :mod:`site_example` — public site template


.. ── Indices ──────────────────────────────────────────────────────────────────


------------------
Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
