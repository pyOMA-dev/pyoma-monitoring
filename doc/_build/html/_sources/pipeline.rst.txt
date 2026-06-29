The Daily Processing Pipeline
==============================

The daily processing loop is coordinated by ``daily.py``, a command-line
tool that runs four stages: file ingestion, statistics, modal analysis, and
plotting.  In production the loop is driven by ``daily2.sh``, a cron wrapper
that iterates over quantities and window durations.

This page documents the orchestration layer.  For the pyOMA-internal per-window
workflow (``PreProcessSignals`` → ``VarSSIRef`` → ``StabilCluster``), see
`pyOMA's continuous monitoring page
<https://py-oma.readthedocs.io/en/latest/continuous_monitoring.html#pyoma-integration>`_.


CLI reference
-------------

.. code-block:: bash

   python daily.py -d 120 -q accel --file_info --stats --modal --plot \
       --tmp_dir=/dev/shm/tower_tmp

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Effect
   * - ``-d <minutes>``
     - Window duration in minutes.  Supported values: 10, 30, 60, 120.
   * - ``-q <quantity>``
     - Measurement quantity.  One of ``accel``, ``wind``, ``temp``,
       ``strain_rosettes``.
   * - ``--file_info``
     - Scan newly arrived files and update the file-info database.
   * - ``--stats``
     - Compute per-channel statistics for every new window.
   * - ``--modal``
     - Run modal analysis (VarSSIRef + StabilCluster) for every valid window.
       Only valid with ``-q accel`` or ``-q strain_rosettes``.
   * - ``--plot``
     - Generate daily trend and waterfall plots and write them to
       ``--tmp_dir``.
   * - ``--tmp_dir=<path>``
     - Directory for temporary files and plots.  Defaults to ``cwd``.
   * - ``--dtstart=YYYY-MM-DD HH:MM``
     - Process only windows at or after this UTC-naive timestamp.  When
       ``--file_info`` is given, ``dtstart`` is derived automatically from
       the first newly arrived file.
   * - ``--loglevel=INFO``
     - Python logging level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).


Stage 1 — File ingestion and quality assessment
-----------------------------------------------

``get_file_info(origin, create_new=True)`` scans new files returned by the
active site's ``file_list_fn`` and records per-file metadata plus per-channel
statistics:

.. code-block:: python

   # Condensed from monitoring.create_file_info
   for file in filelist:
       file_time, file_size, headers, units, start_time, sample_rate, measurement \
           = read_file(file)

       if close_to_utc_transition(file_time):
           continue                        # skip files near DST transition

       duration = datetime.timedelta(seconds=measurement.shape[0] / sample_rate)
       sync_time = get_synchronized_time(start_time, file_time, duration)

       dst['start_time'] = (['time'], [TC.to_posix(start_time)])
       dst['file_time']  = (['time'], [TC.to_posix(file_time)])
       dst.coords['time'] = [TC.to_storage_coord(sync_time)]

       ds = ds.combine_first(dst)

Three things happen here simultaneously:

1. The raw timestamps (``start_time``, ``file_time``) are stored as POSIX
   float seconds — timezone-agnostic and arithmetic-friendly.
2. The synchronised wall-clock time is converted to UTC-naive ``datetime64``
   and used as the ``time`` coordinate (the database index).
3. Quality flags — kurtosis, min == max, out-of-range values — are set per
   channel.  Channels with ``error == True`` are excluded from downstream OMA.

``compute_gap_lengths`` is appended to every file-info dataset on load.  It
computes the gap in samples between consecutive files so that
``get_slice`` can skip windows where files are non-consecutive.


Stage 2 — Signal preprocessing (statistics)
--------------------------------------------

``create_stats`` iterates over fixed-duration windows aligned to the local
Berlin time grid and extracts one signal slice per window:

.. code-block:: python

   # Condensed from monitoring.create_stats
   _aware_iter, time_iter_naive = TC.make_index(dtstart, until, minutes)

   for time_ in time_iterator:
       start_time = TC.to_local(time_)          # stored UTC-naive → Berlin-aware

       data_slice = get_slice_corrected(start_time, duration, quantity, file_info)
       if data_slice is None:
           continue                              # no coverage or gap in files

       this_ds.coords['time'] = [start_time.to_datetime64()]
       process_ds = process_ds.combine_first(this_ds)

``get_slice_corrected`` manages three sub-steps:

1. **Raw slice extraction** (``get_slice``) — selects files covering the
   window, truncates samples at both ends to align with the requested
   boundaries, fills small gaps (≤ 32 samples for strain channels) with
   ``np.nan``, and stacks the result into a single ``(T, N)`` array.
2. **Site transform** — the active site's ``transforms[quantity]`` callback
   is applied.  For ``accel`` no transform is registered; for ``wind`` the
   callback converts the raw Wg/Wr channels to Cartesian components and
   applies a circular-mean correction; for ``strain_rosettes`` it converts
   FBG wavelengths to strain with temperature compensation.
3. **Slice caching** — the corrected slice is saved as a ``.npz`` file
   under ``<slice_root>/<dur>-minutes/slices_<quantity>/YYYY/MM/``.  On
   subsequent runs ``get_slice_corrected`` loads the cached ``.npz``
   directly, bypassing the reader and transform entirely.

This two-level caching (process-local Dataset and on-disk ``.npz``) means
that interrupted runs resume exactly where they left off without
reprocessing already-completed windows.


Stage 3 — Modal analysis
-------------------------

``create_modal_results`` mirrors the statistics loop: it iterates over
error-free windows in the stats database and calls
``modal_analysis_single`` for each.

The per-window OMA workflow is implemented entirely with pyOMA:

.. code-block:: python

   # Condensed from monitoring.modal_analysis_single
   from pyOMA.core.PreProcessingTools import GeometryProcessor, PreProcessSignals
   from pyOMA.core.VarSSIRef import VarSSIRef
   from pyOMA.core.StabilDiagram import StabilCluster

   # 1. Preprocessing
   prep_data = PreProcessSignals(measurement, sample_rate,
                   ref_channels=ref_channels,
                   accel_channels=accel_channels,
                   channel_headers=headers,
                   start_time=start_time)
   prep_data.add_chan_dofs(chan_dofs)
   s_vals_psd = prep_data.sv_psd(1444, method='blackman-tukey', refs_only=False)

   # 2. Identification
   modal_data = VarSSIRef.init_from_config(ssi_file, prep_data)

   # 3. Automated stabilisation
   stabil_calc = StabilCluster(modal_data)
   stabil_calc.calculate_soft_critera_matrices()
   stabil_calc.calculate_stabilization_masks()
   stabil_calc.automatic_clearing()
   stabil_calc.automatic_classification()
   stabil_calc.automatic_selection()

   freqs, damping, shapes, orders, *_ = stabil_calc.return_results()

Each of the three objects (``prep_data``, ``modal_data``, ``stabil_calc``) is
cached to disk as a ``.npz`` file.  If the pipeline is interrupted between
any two of these steps, the next run loads from cache rather than
recomputing, which is important because ``VarSSIRef`` over a 120-minute
window at 100 Hz can take several minutes per window.

The ``setup_prep[quantity]`` hook on the active site provides the channel
role mapping required by ``PreProcessSignals`` (reference channels,
acceleration channels, displacement channels, and the channel-to-DOF
dictionary).  This hook is the only piece of site knowledge that leaks into
the modal analysis; everything else is generic.


Stage 4 — Concurrent write safety (MultiLock)
----------------------------------------------

Both ``create_stats`` and ``create_modal_results`` support parallel execution
across ``num_workers`` processes.  Each worker writes to a private
``<name>.<pid>.nc`` file during the long computation phase, then merges into
the shared master file at the end of a chunk:

.. code-block:: python

   # Condensed from monitoring.save_ds
   with MultiLock(savepath):
       if reload_current:
           current_ds = xr.open_dataset(savepath)
           current_ds.load(); current_ds.close()

       dupes, _, _ = np.intersect1d(new_ds.time, current_ds.time, ...)
       current_ds = current_ds.drop_sel(time=dupes)
       current_ds = current_ds.combine_first(new_ds)

       current_ds.to_netcdf(tempfile)
       os.rename(tempfile, savepath)   # atomic on POSIX filesystems

``MultiLock`` layers a per-process ``.pid.lock`` file on top of
``simpleflock`` to handle a race condition where ``simpleflock`` can briefly
grant the lock to two processes simultaneously.


The cron wrapper (daily2.sh)
-----------------------------

In production, ``daily2.sh`` is called once per day by cron.  It iterates
over quantities and durations, passing the ``--dtstart`` flag derived from the
most recently arrived file to each subsequent duration run:

.. code-block:: bash

   TMPDIR=/dev/shm/tower_tmp
   mkdir ${TMPDIR}

   # 120-minute file ingestion sets DTSTART for shorter-duration runs
   python daily.py -d 120 -q accel --file_info --stats --modal \
       --tmp_dir=${TMPDIR} > ${TMPDIR}/tower_out.txt

   if test -f "${TMPDIR}/dtstart.tmp"; then
       DTSTART=$(cat ${TMPDIR}/dtstart.tmp)
       python daily.py -d  60 -q accel --stats --modal \
           --dtstart=${DTSTART} >> ${TMPDIR}/tower_out.txt
       python daily.py -d  30 -q accel --stats --modal \
           --dtstart=${DTSTART} >> ${TMPDIR}/tower_out.txt
       python daily.py -d  10 -q accel --stats --modal --plot \
           --dtstart=${DTSTART} >> ${TMPDIR}/tower_out.txt
   fi
   # results and plots are emailed to the monitoring team
   cat ${TMPDIR}/tower_out.txt | mail -s "..." \
       -a ${TMPDIR}/modal_accel_10.png your@email.de

The ``--file_info`` flag at 120 minutes writes the earliest new-file
timestamp to ``dtstart.tmp``.  All subsequent shorter-duration runs use that
timestamp so they process exactly the same new data range, rather than
re-scanning the full history.

Quantities without modal analysis (``wind``, ``temp``) run through the same
loop structure but omit the ``--modal`` flag.
