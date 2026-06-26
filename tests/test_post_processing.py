"""Tests for post_processing.py.

Structure
---------
Unit tests (no I/O)
    TestDtstartToUtc    — _dtstart_to_stored_utc() timezone conversion

Integration tests (require on-disk data, marked with @integration)
    TestLoadFilterMerge — load_filter_merge() using result_db_short
    TestPlotDaily       — plot_daily() returns figures without raising
    TestPlotWaterfall   — plot_waterfall() returns a figure without raising

Note on pandas 3.0 and timezone-awareness
------------------------------------------
`load_filter_merge` uses a default `time_range` built from tz-aware
``pd.Timestamp(..., tz='Europe/Berlin')`` objects but applies it to a
tz-naive UTC dataset via ``ds.sel(time=slice(...))``; this raises
``TypeError`` in pandas ≥ 3.0.  All integration tests that call
``load_filter_merge`` therefore supply explicit tz-naive timestamps to
work around the issue, and include a comment documenting the production
code limitation.
"""
import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe for CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from conftest import skip_if_no_data

from post_processing import (
    _dtstart_to_stored_utc,
    load_filter_merge,
    plot_daily,
    plot_waterfall,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tz_naive(ts_str: str) -> pd.Timestamp:
    """Return a tz-naive pd.Timestamp suitable for ``load_filter_merge``."""
    return pd.Timestamp(ts_str)


# ---------------------------------------------------------------------------
# Unit tests — _dtstart_to_stored_utc
# ---------------------------------------------------------------------------

class TestDtstartToUtc:
    """The function mirrors the double-Berlin-localisation that ``create_stats``
    uses when storing time coordinates.  For a naive Berlin local time T:
      stored UTC = (T → Berlin aware → UTC naive → Berlin aware → .to_datetime64())
    The final ``.to_datetime64()`` converts the Berlin-aware timestamp back to UTC,
    so the stored coordinate is offset by *two* Berlin-UTC offsets.
    """

    def test_returns_datetime64_subtype(self):
        dtstart = np.datetime64("2026-06-01T00:00")
        result = _dtstart_to_stored_utc(dtstart)
        # Accept any datetime64 subtype (ns or s depending on pandas/xarray version)
        assert np.issubdtype(result.dtype, np.datetime64)

    def test_summer_midnight_berlin(self):
        # 2026-06-01 00:00 Berlin (CEST = UTC+2):
        #   → UTC naive 2026-05-31T22:00
        #   → re-localize Berlin (CEST) → 2026-05-31T22:00+02:00
        #   → .to_datetime64() (UTC) → 2026-05-31T20:00
        dtstart = np.datetime64("2026-06-01T00:00")
        result = _dtstart_to_stored_utc(dtstart)
        assert result == np.datetime64("2026-05-31T20:00:00")

    def test_winter_midnight_berlin(self):
        # 2026-01-05 00:00 Berlin (CET = UTC+1):
        #   → UTC naive 2026-01-04T23:00
        #   → re-localize Berlin (CET) → 2026-01-04T23:00+01:00
        #   → .to_datetime64() (UTC) → 2026-01-04T22:00
        dtstart = np.datetime64("2026-01-05T00:00")
        result = _dtstart_to_stored_utc(dtstart)
        assert result == np.datetime64("2026-01-04T22:00:00")

    def test_summer_2am_berlin(self):
        # 2026-06-01 02:00 Berlin (CEST):
        #   → UTC naive 2026-05-31T24:00 = 2026-06-01T00:00
        #   → re-localize Berlin → 2026-06-01T00:00+02:00
        #   → .to_datetime64() → 2026-05-31T22:00:00 UTC
        dtstart = np.datetime64("2026-06-01T02:00")
        result = _dtstart_to_stored_utc(dtstart)
        assert result == np.datetime64("2026-05-31T22:00:00")

    def test_deterministic(self):
        """Calling twice with the same input produces the same result."""
        dtstart = np.datetime64("2026-06-10T06:00")
        assert _dtstart_to_stored_utc(dtstart) == _dtstart_to_stored_utc(dtstart)


# ---------------------------------------------------------------------------
# Integration tests — load_filter_merge
# ---------------------------------------------------------------------------

# Tz-naive time range covering a week of data in the short db.
_T0 = _tz_naive("2026-06-05")
_T1 = _tz_naive("2026-06-10")


@skip_if_no_data
class TestLoadFilterMerge:
    """Tests for load_filter_merge using tz-naive time_range to avoid the
    pandas ≥ 3.0 TypeError when filtering a tz-naive dataset with tz-aware
    timestamps (a known limitation of the production code).
    """

    DURATION = pd.Timedelta(minutes=120)
    QUANTITY = "accel"

    def test_returns_tuple_of_2(self, short_db):
        result = load_filter_merge(
            self.QUANTITY, self.DURATION, time_range=(_T0, _T1)
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_dataset_has_modal_variables(self, short_db):
        ds, _ = load_filter_merge(
            self.QUANTITY, self.DURATION, time_range=(_T0, _T1)
        )
        for var in ("frequencies", "damping", "num_modes"):
            assert var in ds, f"Variable {var!r} missing"

    def test_dataset_has_environmental_variables(self, short_db):
        ds, _ = load_filter_merge(
            self.QUANTITY, self.DURATION, time_range=(_T0, _T1)
        )
        assert "wind" in ds
        assert "temp" in ds

    def test_filter_ranges_in_metadata(self, short_db):
        _, ranges = load_filter_merge(
            self.QUANTITY, self.DURATION, time_range=(_T0, _T1)
        )
        _time_range, _rms_range, wind_range, temp_range, f_range = ranges
        assert len(wind_range) == 2
        assert len(temp_range) == 2
        assert len(f_range) == 2

    def test_frequencies_in_default_f_range(self, short_db):
        """All returned frequencies fall within the default mode_pair=0 range (0–4 Hz)."""
        ds, (_, _, _, _, f_range) = load_filter_merge(
            self.QUANTITY, self.DURATION,
            time_range=(_T0, _T1),
            mode_pair=0,
        )
        freqs = ds["frequencies"].values
        valid = freqs[~np.isnan(freqs)]
        if len(valid) > 0:
            assert (valid >= f_range[0]).all()
            assert (valid <= f_range[1]).all()

    def test_wind_filter_mode_1(self, short_db):
        """After wind_range=1 (weak wind, ≤ 5.4 m/s), all wind values ≤ upper bound."""
        ds, (_, _, wind_range, _, _) = load_filter_merge(
            self.QUANTITY, self.DURATION,
            time_range=(_T0, _T1),
            wind_range=1,
        )
        wind = ds["wind"].values
        valid = wind[~np.isnan(wind)]
        if len(valid) > 0:
            assert (valid <= wind_range[1]).all()

    def test_time_range_filter_tz_naive(self, short_db):
        """Time-range filter with tz-naive timestamps reduces the dataset correctly."""
        t_start = _tz_naive("2026-06-07")
        t_end = _tz_naive("2026-06-09")
        ds, _ = load_filter_merge(
            self.QUANTITY, self.DURATION,
            time_range=(t_start, t_end),
        )
        times = ds["time"].values
        assert (times >= t_start.to_datetime64()).all()
        assert (times <= t_end.to_datetime64()).all()


# ---------------------------------------------------------------------------
# Integration tests — plot_daily
# ---------------------------------------------------------------------------

@skip_if_no_data
class TestPlotDaily:
    """Smoke tests: plot_daily must return figures and not raise."""

    DURATION = pd.Timedelta(minutes=120)
    DTSTART = np.datetime64("2026-06-01T00:00")

    def _close_all(self):
        plt.close("all")

    def test_plot_daily_accel_returns_tuple(self, short_db):
        try:
            result = plot_daily("accel", self.DURATION, self.DTSTART)
            assert isinstance(result, tuple)
            fig1, _fig2 = result
            assert isinstance(fig1, plt.Figure)
        finally:
            self._close_all()

    def test_plot_daily_accel_has_second_figure(self, short_db):
        """For modal quantities plot_daily should return a second scatter figure."""
        try:
            _, fig2 = plot_daily("accel", self.DURATION, self.DTSTART)
            assert fig2 is not None
            assert isinstance(fig2, plt.Figure)
        finally:
            self._close_all()

    def test_plot_daily_wind_returns_tuple(self, short_db):
        try:
            result = plot_daily("wind", self.DURATION, self.DTSTART)
            assert isinstance(result, tuple)
            fig1, fig2 = result
            assert isinstance(fig1, plt.Figure)
            assert fig2 is None  # non-modal quantity has no scatter figure
        finally:
            self._close_all()

    def test_plot_daily_temp_returns_tuple(self, short_db):
        try:
            result = plot_daily("temp", self.DURATION, self.DTSTART)
            assert isinstance(result, tuple)
            fig1, _ = result
            assert isinstance(fig1, plt.Figure)
        finally:
            self._close_all()

    def test_plot_daily_does_not_raise_on_empty_range(self, short_db):
        """dtstart past all data → empty but should not raise."""
        dtstart_late = np.datetime64("2026-12-01T00:00")
        try:
            result = plot_daily("wind", self.DURATION, dtstart_late)
            assert isinstance(result, tuple)
        finally:
            self._close_all()


# ---------------------------------------------------------------------------
# Integration tests — plot_waterfall
# ---------------------------------------------------------------------------

@skip_if_no_data
class TestPlotWaterfall:

    DURATION = pd.Timedelta(minutes=120)
    DTSTART = np.datetime64("2026-06-01T00:00")

    def test_returns_figure(self, short_db):
        try:
            fig = plot_waterfall("accel", self.DURATION, self.DTSTART)
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close("all")

    def test_does_not_raise_on_empty_range(self, short_db):
        dtstart_late = np.datetime64("2026-12-01T00:00")
        try:
            fig = plot_waterfall("accel", self.DURATION, dtstart_late)
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close("all")
