"""Unit tests for time_convention.TimeConvention.

Each test method pins the exact values established in test_time_handling.py,
verifying that the class encapsulates the correct conversions.
"""
import datetime

import numpy as np
import pandas as pd
import pytz
import xarray as xr

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from time_convention import TimeConvention

berlin = pytz.timezone("Europe/Berlin")
TC = TimeConvention()


def _berlin(year, month, day, hour=0, minute=0):
    return berlin.localize(datetime.datetime(year, month, day, hour, minute))


# ---------------------------------------------------------------------------
# to_storage_coord — mirrors TestTzConvertChain
# ---------------------------------------------------------------------------

class TestToStorageCoord:

    def test_summer_midnight_stored_as_utc_minus2(self):
        dt = _berlin(2026, 6, 1, 0, 0)
        coord = TC.to_storage_coord(dt)
        assert np.datetime64(coord, "s") == np.datetime64("2026-05-31T22:00:00", "s")

    def test_winter_midnight_stored_as_utc_minus1(self):
        dt = _berlin(2026, 1, 5, 0, 0)
        coord = TC.to_storage_coord(dt)
        assert np.datetime64(coord, "s") == np.datetime64("2026-01-04T23:00:00", "s")

    def test_result_is_tz_naive_datetime64(self):
        dt = _berlin(2026, 6, 1, 12, 0)
        coord = TC.to_storage_coord(dt)
        assert np.issubdtype(coord.dtype, np.datetime64)

    def test_consistent_with_posix_roundtrip(self):
        dt = _berlin(2026, 6, 15, 8, 30)
        chain = TC.to_storage_coord(dt)
        posix = dt.timestamp()
        via_posix = datetime.datetime.fromtimestamp(posix, tz=pytz.UTC)
        posix_coord = pd.Timestamp(via_posix).tz_localize(None).to_datetime64()
        assert np.datetime64(chain, "s") == np.datetime64(posix_coord, "s")


# ---------------------------------------------------------------------------
# to_local — the "double localisation" pattern
# ---------------------------------------------------------------------------

class TestToLocal:

    def test_summer_utc_naive_berlin_localized_shifts_back_two_hours(self):
        # 22:00 UTC-naive localized as Berlin (CEST=+2) → 22:00+02 → to_datetime64 = 20:00 UTC
        stored = np.datetime64("2026-05-31T22:00:00", "ns")
        result = TC.to_local(stored)
        assert result.tzinfo is not None
        assert np.datetime64(result.to_datetime64(), "s") == np.datetime64("2026-05-31T20:00:00", "s")

    def test_time_in_spring_gap_returns_nat(self):
        # 02:30 on spring-forward night doesn't exist in Berlin
        stored = pd.Timestamp("2026-03-29 02:30:00")
        result = TC.to_local(stored)
        assert pd.isnull(result)

    def test_normal_time_returns_aware_timestamp(self):
        stored = pd.Timestamp("2026-06-01 10:00:00")
        result = TC.to_local(stored)
        assert result.tzinfo is not None

    def test_winter_naive_localized_correctly(self):
        stored = np.datetime64("2026-01-04T23:00:00", "ns")
        result = TC.to_local(stored)
        assert result.tzinfo is not None


# ---------------------------------------------------------------------------
# posix_to_datetime64 — mirrors TestPosixTimestampStorage
# ---------------------------------------------------------------------------

class TestPosixToDatetime64:

    def test_scalar_posix_recovers_midnight_utc(self):
        dt = _berlin(2026, 6, 1, 0, 0)
        posix = dt.timestamp()
        recovered = TC.posix_to_datetime64(posix)
        expected = pd.Timestamp(dt).tz_convert("UTC").tz_localize(None).to_datetime64()
        assert np.datetime64(recovered, "s") == np.datetime64(expected, "s")

    def test_xarray_float_dataarray_roundtrip(self):
        dt = _berlin(2026, 6, 1, 0, 0)
        posix = dt.timestamp()
        da = xr.DataArray([posix], dims="time")
        recovered = TC.posix_to_datetime64(da)
        assert recovered.dtype == np.dtype("datetime64[s]")
        expected = pd.Timestamp(dt).tz_convert("UTC").tz_localize(None).to_datetime64()
        assert np.datetime64(recovered.values[0], "s") == np.datetime64(expected, "s")

    def test_result_dtype_is_datetime64_s(self):
        da = xr.DataArray([1748736000.0], dims="time")
        result = TC.posix_to_datetime64(da)
        assert result.dtype == np.dtype("datetime64[s]")


# ---------------------------------------------------------------------------
# to_posix
# ---------------------------------------------------------------------------

class TestToPosit:

    def test_summer_dst_posix_differs_from_winter(self):
        summer = _berlin(2026, 7, 1, 2, 0)
        winter = _berlin(2026, 1, 7, 2, 0)
        assert TC.to_posix(summer) != TC.to_posix(winter)

    def test_to_posix_is_utc_based(self):
        dt = _berlin(2026, 6, 1, 0, 0)
        posix = TC.to_posix(dt)
        recovered = datetime.datetime.fromtimestamp(posix, tz=pytz.UTC)
        assert recovered.hour == 22  # Berlin midnight = 22:00 UTC in CEST


# ---------------------------------------------------------------------------
# make_index — mirrors TestRruleChain
# ---------------------------------------------------------------------------

class TestMakeIndex:

    def test_summer_midnight_maps_to_22h_utc(self):
        _, naive = TC.make_index(datetime.datetime(2026, 6, 1, 0, 0),
                                 until=datetime.datetime(2026, 6, 1, 4, 0), minutes=120)
        assert naive[0] == np.datetime64("2026-05-31T22:00:00", "ns")

    def test_step_is_120_minutes_in_utc(self):
        _, naive = TC.make_index(datetime.datetime(2026, 6, 1, 0, 0),
                                 until=datetime.datetime(2026, 6, 1, 6, 0), minutes=120)
        diffs = np.diff(naive).astype("timedelta64[m]")
        for diff in diffs:
            assert diff == np.timedelta64(120, "m")

    def test_naive_array_dtype_is_datetime64ns(self):
        _, naive = TC.make_index(datetime.datetime(2026, 6, 1, 0, 0),
                                 until=datetime.datetime(2026, 6, 1, 4, 0), minutes=120)
        assert naive.dtype == np.dtype("datetime64[ns]")

    def test_aware_list_has_berlin_tz(self):
        aware, _ = TC.make_index(datetime.datetime(2026, 6, 1, 0, 0),
                                  until=datetime.datetime(2026, 6, 1, 4, 0), minutes=120)
        assert all(ts.tzinfo is not None for ts in aware)

    def test_winter_chain_maps_to_23h_utc(self):
        _, naive = TC.make_index(datetime.datetime(2026, 1, 5, 0, 0),
                                  until=datetime.datetime(2026, 1, 5, 4, 0), minutes=120)
        assert naive[0] == np.datetime64("2026-01-04T23:00:00", "ns")


# ---------------------------------------------------------------------------
# is_near_dst_transition — mirrors TestDstGapLocalize / close_to_utc_transition
# ---------------------------------------------------------------------------

class TestIsNearDstTransition:

    def test_time_near_known_transition_returns_true(self):
        dt = _berlin(2016, 3, 27, 1, 0)
        assert TC.is_near_dst_transition(dt) is True

    def test_time_far_from_transition_returns_false(self):
        dt = _berlin(2026, 6, 15, 12, 0)
        assert TC.is_near_dst_transition(dt) is False

    def test_default_window_is_3_hours(self):
        # The transition at 2016-03-27 01:00 UTC; querying 04:01 should be false
        # 2016-03-27 04:01 local ≈ 02:01 UTC → 3h01m after transition → outside
        dt = _berlin(2016, 3, 27, 5, 5)
        assert TC.is_near_dst_transition(dt) is False

    def test_custom_hours_window(self):
        # The transition at 2016-03-27 01:00 UTC; 2 hours after = just outside 1h window
        dt = _berlin(2016, 3, 27, 4, 30)  # ≈ 2.5h after transition
        assert TC.is_near_dst_transition(dt, hours=1) is False
        assert TC.is_near_dst_transition(dt, hours=4) is True


# ---------------------------------------------------------------------------
# gap_lengths — fixes Bug 2 (NaT sentinel) and uses 's' unit (Bug 1 fix)
# ---------------------------------------------------------------------------

class TestGapLengths:

    def _make_ds(self, n_files=4, gap_seconds=0.0):
        """Create a synthetic file_info-like Dataset."""
        # Start times as POSIX seconds: hourly files
        base_posix = pd.Timestamp("2026-06-01 00:00:00", tz="UTC").timestamp()
        sample_rate = 100.0
        duration = 3600.0
        start_posix = [base_posix + i * (duration + gap_seconds) for i in range(n_files)]
        times = pd.date_range("2026-06-01", periods=n_files, freq="h")
        return xr.Dataset(
            {
                "start_time": ("time", start_posix),
                "duration":   ("time", [duration] * n_files),
                "sample_rate":("time", [sample_rate] * n_files),
            },
            coords={"time": times},
        )

    def test_last_element_is_nan_not_sentinel(self):
        ds = self._make_ds()
        gaps = TC.gap_lengths(ds["start_time"], ds["duration"], ds["sample_rate"])
        assert np.isnan(gaps[-1])

    def test_last_element_is_not_large_negative(self):
        ds = self._make_ds()
        gaps = TC.gap_lengths(ds["start_time"], ds["duration"], ds["sample_rate"])
        assert not (gaps[-1] < -1e6)

    def test_consecutive_files_have_zero_gap(self):
        ds = self._make_ds(gap_seconds=0.0)
        gaps = TC.gap_lengths(ds["start_time"], ds["duration"], ds["sample_rate"])
        for g in gaps[:-1]:
            assert abs(g) < 2, f"Expected ~0 gap, got {g}"

    def test_known_gap_in_samples(self):
        # 10-second gap between files at 100 Hz → 1000 samples
        ds = self._make_ds(n_files=3, gap_seconds=10.0)
        gaps = TC.gap_lengths(ds["start_time"], ds["duration"], ds["sample_rate"])
        assert abs(gaps[0] - 1000.0) < 2  # allow ±2 samples rounding

    def test_duration_unit_is_seconds_not_microseconds(self):
        # Bug 1 regression: if 'us' were used, gap for a 1h file would be wrong by ~3600000x
        ds = self._make_ds(gap_seconds=0.0)
        gaps = TC.gap_lengths(ds["start_time"], ds["duration"], ds["sample_rate"])
        # All interior gaps should be close to 0, not ≈ 3.6e9 samples
        for g in gaps[:-1]:
            assert abs(g) < 100, f"Suspicious gap magnitude suggests 'us' unit bug: {g}"
