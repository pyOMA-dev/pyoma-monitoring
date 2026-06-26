# pyOMA-Monitoring

Long-term structural health monitoring (SHM) pipeline for a **guyed mast**
(~188 m tall), developed at Bauhaus-Universität Weimar as part of
doctoral research on ambient-vibration-based system identification.

The pipeline reads multi-sensor binary data recorded by a Gantner Q.station
controller, computes signal statistics and Operational Modal Analysis (OMA)
via [pyOMA](https://github.com/simonmarwitz/pyOMA), and stores results in
NetCDF/xarray databases. A daily cron job orchestrates the full run and
e-mails result plots to the project team.

---

## Instrumentation overview

| Sensor type | Model | Positions |
|---|---|---|
| Accelerometers | PCB 393A03 (1 V/g), PCB 393B31 (10 V/g) | +188 m, +163.9 m, +141.9 m, +126.9 m, +107.5 m |
| Anemometers (3D ultrasonic) | Thies 3D Ultrasonic Anemometer | +188 m (top), +108.9 m (mid) |
| Temperature (Pt100) | Standard Pt100 RTD | +188 m, +108.9 m, ground box |
| FBG strain rosettes | Fibre Bragg Grating, 3-axis | Guy-wire attachment points (A–D) |
| FBG bolt gauges | Fibre Bragg Grating | Bolted connections (8, 9, 10) |

Raw data is logged continuously by the Gantner controller in proprietary
binary `.dat` format (Gantner UDBF) and transferred daily to the analysis
server via SFTP.

---

## Architecture

```
Gantner Q.station controller
        │  (Gantner UDBF .dat / Labview .bin, daily SFTP transfer)
        ▼
fbg_strain_reader.py / gantner_reader.py      ← low-level binary readers
        │
        ▼
site_geyer.py  ──► config.py                  ← tower-specific configuration
        │            (paths, channels, ranges)
        │  registers Site dataclass in monitoring engine
        ▼
monitoring.py  (generic engine)
  ├── get_file_info()   → accel/wind/temp/strain file-info NetCDF database
  ├── get_stats()       → per-slice statistics NetCDF database
  └── get_modal_results() → SSI-based modal results NetCDF database
        │
        ▼
post_processing.py                 ← Matplotlib plots (daily, waterfall)
        │
        ▼
daily.py  (CLI entry-point, called by daily2.sh cron wrapper)
```

The engine (`monitoring.py`) is site-agnostic — it never imports `config.py`
directly. All site-specific knowledge (paths, channel lists, file-name patterns,
signal transforms, sync policy) is encapsulated in the `Site` dataclass and
registered by `site_geyer.py` at import time via `register_site()` /
`set_active_site()`. To adapt the pipeline to a different monitoring site,
write a new `site_<name>.py` and import it instead.

### Pipeline stages

| Stage | Flag | What it does |
|---|---|---|
| `--file_info` | `get_file_info()` | Scans the raw data directory, builds/updates a file-info database (file paths, timestamps, durations) |
| `--stats` | `get_stats()` | Extracts 30-minute slices, computes RMS, mean, peak and stores them in the stats database |
| `--modal` | `get_modal_results()` | Runs SSI-Cov (via pyOMA) on each 30-minute acceleration slice and stores identified frequencies, damping ratios and mode shapes |
| `--plot` | `plot_daily()` / `plot_waterfall()` | Generates time-history and waterfall plots from the databases |

---

## Installation

```bash
pip install -r requirements.txt
```

`pyOMA` is not on PyPI — install it directly from GitHub:

```bash
pip install git+https://github.com/simonmarwitz/pyOMA.git
```

Python ≥ 3.9 is required.

---

## Configuration

All tower-specific settings live in `config.py`. They are read by `site_geyer.py`
at import time and stored in the `Site` dataclass that is registered with the
engine. Before using the pipeline on a new deployment, update `config.py` with
the correct values:

| Variable | Description |
|---|---|
| `file_root_path` | Root directory where raw `.dat` / `.bin` files are stored |
| `slice_root_path` | Scratch directory for temporary data slices |
| `db_root_path` | Directory where NetCDF result databases are written |
| `modal_conf_dir` | Directory containing SSI configuration files (pole selection etc.) |
| `origins` | Dict mapping each quantity name to its origin tag |
| `subpaths` | Dict mapping each origin tag to its subdirectory under `file_root_path` |
| `all_channels` | Dict mapping each quantity to its primary channel names |
| `optional_channels` | Dict of channels that may or may not be present in a file |
| `ranges` | Physical plausibility limits per channel (used for outlier rejection) |
| `dtstarts` | First valid timestamp for each origin |

To port the pipeline to a **different monitoring site**, create a new
`site_<name>.py` that builds a `monitoring.Site` dataclass from that site's
configuration and calls `monitoring.register_site()` / `monitoring.set_active_site()`.
The engine itself requires no changes.

A `config.example.py` with dummy values is planned (Priority 3).

---

## Usage

```bash
# Typical daily run — acceleration, 120-minute look-back, all stages
python daily.py -d 120 -q accel --file_info --stats --modal --plot --tmp_dir=/tmp/shm

# Options
#   -d <int>      Look-back duration in minutes: 10, 30, 60, or 120
#   -q <str>      Quantity: accel | wind | temp | strain_rosettes
#   --file_info   Update the file-info database
#   --stats       Compute and store statistics
#   --modal       Run OMA (only valid for accel and strain_rosettes)
#   --plot        Generate and save plots to --tmp_dir
#   --tmp_dir=    Directory for temporary files and output plots
#   --dtstart=    Override start datetime (YYYY-MM-DD HH:MM)
```

The cron wrapper `daily2.sh` loops over all three quantities (temp, wind,
accel) and e-mails the output plots. Configure the `RECIPIENTS` variable at
the top of that file before use.

---

## Data format

Raw measurement data is **not included** in this repository (proprietary
binary format, large file sizes). The pipeline expects the following directory
layout under `file_root_path`:

```
file_root_path/
├── towerdata/        # Gantner .dat files (accel, wind, temp)
│   └── YYYY/MM/DD/
└── strain_data/      # Labview .bin files (FBG strain)
    └── YYYY/MM/DD/
```

Files are read by `fbg_strain_reader.py` (Labview binary) and `gantner_reader.py`
(Gantner UDBF `.dat`).

---

## Citation / References

If you use this code, please cite the underlying pyOMA library:

> Marwitz, S. (2024). *pyOMA — Operational Modal Analysis in Python*.
> Bauhaus-Universität Weimar.
> <https://github.com/simonmarwitz/pyOMA>

Relevant publications describing the tower monitoring campaign will be
listed here once publicly available.

---

## License

MIT — see [LICENSE](LICENSE).
