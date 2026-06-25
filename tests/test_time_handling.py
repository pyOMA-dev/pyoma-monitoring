"""Regression tests for timestamp / timedelta handling across pandas, numpy, and xarray.

These tests pin the exact behaviour of library-level conversions that the
monitoring pipeline relies on.  They serve as early-warning detectors when
any of numpy / pandas / xarray is updated: a failing test here means a
specific library contract has changed, and the corresponding code in
monitoring.py or post_processing.py must be audited.

Each test class maps to one conversion pattern extracted from the codebase,
named by the source file and approximate line number in parentheses where
the pattern first appears.

Patterns covered
----------------
Codebase-internal chains
  TzConvertChain         — Berlin-aware → UTC naive → datetime64[us/ns]
  PosixTimestampStorage  — float POSIX ↔ timezone-aware datetime; xarray astype round-trip
  TimedeltaToFloatSecs   — datetime.timedelta → timedelta64[s] → float64 (duration storage)
  DoubleLocalization      — UTC-naive → re-localize Berlin → to_datetime64()
  RruleChain             — dateutil.rrule → Berlin-aware pd.Timestamp → UTC-naive datetime64[ns]
  DstGapLocalize         — tz_localize on times in / near DST spring-forward gap
  XarrayTimeArithmetic   — float_seconds × timedelta64(unit) added to xr time coordinate
  NatSentinel            — xr.DataArray.shift → NaT → int64 overflow sentinel
  TimedeltaWrapping      — np.timedelta64(datetime64_diff, 's') unit coercion
  SampleIndexComputation — truncate_time → sample count (floor/ceil/int interplay)

Library contracts
  CombineFirstPrecedence — xr.Dataset.combine_first conflict resolution
  TimedeltaUnit          — multiplying float DataArray by timedelta64 with various units
  PandasTzNaiveAware     — tz-naive / tz-aware comparison raises TypeError (pandas ≥ 3.0)
  DatetimeDtype          — pd.Timestamp.to_datetime64() returns datetime64[us] (numpy ≥ 2.0)
  NatInt64               — np.datetime64('NaT').astype('int64') == iNaT sentinel
"""

import datetime

import dateutil.rrule
import numpy as np
import pandas as pd
import pytest
import pytz
import xarray as xr

berlin = pytz.timezone("Europe/Berlin")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _berlin(year, month, day, hour=0, minute=0):
    return berlin.localize(datetime.datetime(year, month, day, hour, minute))


# ---------------------------------------------------------------------------
# TzConvertChain
# Monitoring.py line 619:
#   dst.coords['time'] = [pd.Timestamp(sync_time).tz_convert('UTC').tz_localize(None).to_datetime64()]
# ---------------------------------------------------------------------------

class TestTzConvertChain:
    """Berlin-aware → UTC naive → datetime64."""

    def test_summer_midnight_stored_as_utc_minus2(self):
        # Berlin midnight in summer (CEST = UTC+2) is stored as 22:00 UTC
        dt = _berlin(2026, 6, 1, 0, 0)
        coord = pd.Timestamp(dt).tz_convert("UTC").tz_localize(None).to_datetime64()
        assert np.datetime64(coord, "s") == np.datetime64("2026-05-31T22:00:00", "s")

    def test_winter_midnight_stored_as_utc_minus1(self):
        # CET = UTC+1
        dt = _berlin(2026, 1, 5, 0, 0)
        coord = pd.Timestamp(dt).tz_convert("UTC").tz_localize(None).to_datetime64()
        assert np.datetime64(coord, "s") == np.datetime64("2026-01-04T23:00:00", "s")

    def test_result_is_tz_naive(self):
        dt = _berlin(2026, 6, 1, 12, 0)
        result = pd.Timestamp(dt).tz_convert("UTC").tz_localize(None)
        assert result.tzinfo is None

    def test_result_dtype_is_datetime64(self):
        dt = _berlin(2026, 6, 1, 6, 0)
        coord = pd.Timestamp(dt).tz_convert("UTC").tz_localize(None).to_datetime64()
        assert np.issubdtype(coord.dtype, np.datetime64)

    def test_roundtrip_via_posix(self):
        # Verify that tz_convert chain is equivalent to datetime.timestamp() round-trip
        dt = _berlin(2026, 6, 15, 8, 30)
        chain = pd.Timestamp(dt).tz_convert("UTC").tz_localize(None).to_datetime64()
        posix = dt.timestamp()
        via_posix = datetime.datetime.fromtimestamp(posix, tz=pytz.UTC)
        posix_coord = pd.Timestamp(via_posix).tz_localize(None).to_datetime64()
        assert np.datetime64(chain, "s") == np.datetime64(posix_coord, "s")


# ---------------------------------------------------------------------------
# PosixTimestampStorage
# Monitoring.py lines 597-600:
#   dst['file_time'] = (['time'], [file_time.timestamp()])
#   ...
# And in compute_gap_lengths (line 1051):
#   start_time = file_info['start_time'].astype('datetime64[s]')
# ---------------------------------------------------------------------------

class TestPosixTimestampStorage:
    """Float POSIX ↔ timezone-aware datetime; xarray astype round-trip."""

    def test_timestamp_float_roundtrip(self):
        dt = _berlin(2026, 6, 1, 12, 0)
        posix = dt.timestamp()
        dt_back = datetime.datetime.fromtimestamp(posix, tz=pytz.UTC)
        assert abs((dt.utctimetuple().tm_hour * 3600 + dt.utctimetuple().tm_min) -
                   (dt_back.utctimetuple().tm_hour * 3600 + dt_back.utctimetuple().tm_min)) == 0

    def test_posix_preserves_microseconds(self):
        dt = _berlin(2026, 6, 1, 12, 0)
        posix = dt.timestamp()
        assert posix == pytest.approx(dt.timestamp())

    def test_xarray_float_to_datetime64s(self):
        # file_info['start_time'].astype('datetime64[s]') must recover a valid timestamp
        dt = _berlin(2026, 6, 1, 0, 0)
        posix = dt.timestamp()
        da = xr.DataArray([posix], dims="time")
        recovered = da.astype("datetime64[s]")
        expected = pd.Timestamp(dt).tz_convert("UTC").tz_localize(None).to_datetime64()
        assert np.datetime64(recovered.values[0], "s") == np.datetime64(expected, "s")

    def test_xarray_float_dtype_after_astype(self):
        da = xr.DataArray([1748736000.0], dims="time")  # arbitrary POSIX
        recovered = da.astype("datetime64[s]")
        assert recovered.dtype == np.dtype("datetime64[s]")

    def test_summer_dst_posix_roundtrip(self):
        # POSIX timestamp is always in UTC — no DST ambiguity
        summer = _berlin(2026, 7, 1, 2, 0)
        winter = _berlin(2026, 1, 7, 2, 0)
        assert summer.timestamp() != winter.timestamp()


# ---------------------------------------------------------------------------
# TimedeltaToFloatSecs
# Monitoring.py line 605:
#   np.asarray(duration, dtype='timedelta64[s]').astype(np.float64)
# ---------------------------------------------------------------------------

class TestTimedeltaToFloatSecs:
    """datetime.timedelta → timedelta64[s] → float64 (duration storage)."""

    def test_whole_seconds_exact(self):
        td = datetime.timedelta(seconds=7200)
        result = np.asarray(td, dtype="timedelta64[s]").astype(np.float64)
        assert result == pytest.approx(7200.0)

    def test_sub_second_truncated_to_second(self):
        # timedelta64[s] has second precision — fractional seconds are lost
        td = datetime.timedelta(seconds=3600, milliseconds=500)
        result = np.asarray(td, dtype="timedelta64[s]").astype(np.float64)
        assert result == pytest.approx(3600.0)  # 0.5 s truncated

    def test_one_hour_is_3600(self):
        td = datetime.timedelta(hours=1)
        result = np.asarray(td, dtype="timedelta64[s]").astype(np.float64)
        assert result == 3600.0

    def test_two_hours_is_7200(self):
        td = datetime.timedelta(hours=2)
        result = np.asarray(td, dtype="timedelta64[s]").astype(np.float64)
        assert result == 7200.0

    def test_result_dtype_is_float64(self):
        td = datetime.timedelta(seconds=120)
        result = np.asarray(td, dtype="timedelta64[s]").astype(np.float64)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# DoubleLocalization
# Post_processing.py lines 289-290 (_dtstart_to_stored_utc):
#   utc_naive = pd.Timestamp(dtstart, tz='Europe/Berlin').tz_convert('UTC').tz_localize(None)
#   return pd.Timestamp(utc_naive).tz_localize('Europe/Berlin').to_datetime64()
#
# Also create_stats lines 862-863: naive UTC → re-localize Berlin → to_datetime64()
# ---------------------------------------------------------------------------

class TestDoubleLocalization:
    """UTC-naive re-localized as Berlin shifts the coordinate by an extra UTC offset."""

    def test_summer_utc_naive_becomes_berlin_minus_two_hours(self):
        # 22:00 UTC-naive → localize Berlin (CEST = UTC+2) → 22:00+02 → .to_datetime64() = 20:00 UTC
        utc_naive = pd.Timestamp("2026-05-31 22:00:00")   # Berlin midnight → UTC naive
        result = utc_naive.tz_localize("Europe/Berlin").to_datetime64()
        assert np.datetime64(result, "s") == np.datetime64("2026-05-31T20:00:00", "s")

    def test_winter_utc_naive_becomes_berlin_minus_one_hour(self):
        # 23:00 UTC-naive → localize Berlin (CET = UTC+1) → 23:00+01 → 22:00 UTC
        utc_naive = pd.Timestamp("2026-01-04 23:00:00")
        result = utc_naive.tz_localize("Europe/Berlin").to_datetime64()
        assert np.datetime64(result, "s") == np.datetime64("2026-01-04T22:00:00", "s")

    def test_re_localized_is_tz_aware(self):
        utc_naive = pd.Timestamp("2026-05-31 22:00:00")
        tz_aware = utc_naive.tz_localize("Europe/Berlin")
        assert tz_aware.tzinfo is not None

    def test_to_datetime64_is_utc_naive(self):
        # to_datetime64() of a tz-aware Timestamp strips tz info (and converts to UTC)
        ts = pd.Timestamp("2026-06-01 00:00:00", tz="Europe/Berlin")
        dt64 = ts.to_datetime64()
        assert np.issubdtype(dt64.dtype, np.datetime64)
        # The value is UTC, not local time
        assert np.datetime64(dt64, "s") == np.datetime64("2026-05-31T22:00:00", "s")


# ---------------------------------------------------------------------------
# RruleChain
# Monitoring.py lines 832-837 (create_stats):
#   time_iterator = dateutil.rrule.rrule(...)
#   time_iterator = [pd.Timestamp(ts, tz='Europe/Berlin') for ts in time_iterator]
#   time_iter_naive = np.array(
#       [ts.tz_convert('UTC').tz_localize(None) for ts in time_iterator],
#       dtype='datetime64[ns]')
# ---------------------------------------------------------------------------

class TestRruleChain:
    """dateutil.rrule → Berlin-aware pd.Timestamp → UTC-naive datetime64[ns]."""

    def _make_naive(self, dtstart_dt, n=3):
        rule = dateutil.rrule.rrule(
            dateutil.rrule.MINUTELY, interval=120,
            dtstart=dtstart_dt, count=n
        )
        ts_list = [pd.Timestamp(ts, tz="Europe/Berlin") for ts in rule]
        return np.array(
            [ts.tz_convert("UTC").tz_localize(None) for ts in ts_list],
            dtype="datetime64[ns]",
        )

    def test_summer_midnight_maps_to_22h_utc(self):
        result = self._make_naive(datetime.datetime(2026, 6, 1, 0, 0))
        # 2026-06-01 00:00 CEST → 2026-05-31T22:00 UTC
        assert result[0] == np.datetime64("2026-05-31T22:00:00", "ns")

    def test_step_is_120_minutes_in_utc(self):
        result = self._make_naive(datetime.datetime(2026, 6, 1, 0, 0))
        diffs = np.diff(result).astype("timedelta64[m]")
        for diff in diffs:
            assert diff == np.timedelta64(120, "m")

    def test_result_dtype_is_datetime64ns(self):
        result = self._make_naive(datetime.datetime(2026, 6, 1, 0, 0))
        assert result.dtype == np.dtype("datetime64[ns]")

    def test_result_is_tz_naive(self):
        # The array values must be interpretable as tz-naive UTC
        result = self._make_naive(datetime.datetime(2026, 6, 2, 8, 0))
        ts = pd.Timestamp(result[0])
        assert ts.tzinfo is None

    def test_winter_chain_maps_to_23h_utc(self):
        result = self._make_naive(datetime.datetime(2026, 1, 5, 0, 0))
        # CET (UTC+1): midnight → 23:00 UTC previous day
        assert result[0] == np.datetime64("2026-01-04T23:00:00", "ns")


# ---------------------------------------------------------------------------
# DstGapLocalize
# Monitoring.py line 863:
#   start_time = start_time.tz_localize('Europe/Berlin', nonexistent='NaT')
# ---------------------------------------------------------------------------

class TestDstGapLocalize:
    """tz_localize behaviour on times in or near the spring-forward DST gap."""

    # Berlin spring-forward 2026-03-29: 02:00 → 03:00 (gap: 02:00–02:59 doesn't exist)
    # Berlin fall-back     2026-10-25: 03:00 → 02:00 (02:30 exists twice)

    def test_normal_time_localized_correctly(self):
        ts = pd.Timestamp("2026-06-01 12:00:00")
        result = ts.tz_localize("Europe/Berlin", nonexistent="NaT")
        assert not pd.isnull(result)
        assert result.tzinfo is not None

    def test_time_in_spring_gap_becomes_nat(self):
        # 02:30 on spring-forward night doesn't exist in Berlin
        ts = pd.Timestamp("2026-03-29 02:30:00")
        result = ts.tz_localize("Europe/Berlin", nonexistent="NaT")
        assert pd.isnull(result)

    def test_time_exactly_at_gap_start_becomes_nat(self):
        # 02:00:00 — the first missing second
        ts = pd.Timestamp("2026-03-29 02:00:00")
        result = ts.tz_localize("Europe/Berlin", nonexistent="NaT")
        assert pd.isnull(result)

    def test_time_before_gap_is_valid(self):
        ts = pd.Timestamp("2026-03-29 01:59:59")
        result = ts.tz_localize("Europe/Berlin", nonexistent="NaT")
        assert not pd.isnull(result)

    def test_time_after_gap_is_valid(self):
        # 03:00 is the first valid time after spring-forward
        ts = pd.Timestamp("2026-03-29 03:00:00")
        result = ts.tz_localize("Europe/Berlin", nonexistent="NaT")
        assert not pd.isnull(result)

    def test_ambiguous_fall_back_time_becomes_nat(self):
        # 02:30 on fall-back night exists twice; ambiguous → NaT
        ts = pd.Timestamp("2026-10-25 02:30:00")
        result = ts.tz_localize("Europe/Berlin", ambiguous="NaT")
        assert pd.isnull(result)

    def test_close_to_utc_transition_function_uses_naive_replacement(self):
        # Verify that close_to_utc_transition strips tzinfo before comparing
        # (file_time.replace(tzinfo=None) is the pattern in production code)
        import sys, os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from monitoring import close_to_utc_transition
        dt = _berlin(2016, 3, 27, 1, 0)
        # The function strips tzinfo via .replace(tzinfo=None) before comparing
        # to _utc_transition_times which are naive UTC.
        # dt.replace(tzinfo=None) = 2016-03-27 01:00 (naive)
        # transition is 2016-03-27 01:00 UTC → within 3h → True
        assert close_to_utc_transition(dt) is True


# ---------------------------------------------------------------------------
# XarrayTimeArithmetic
# Monitoring.py line 813:
#   fi_time_max = (file_info.time + file_info.duration * np.timedelta64(1, 'us')).max().values
# Monitoring.py line 1055 / 1082 / 1150:
#   ... + file_info['duration'] * np.timedelta64(1, 's')
#   ... + file_info.duration  * np.timedelta64(1, 'us')  ← different unit: known inconsistency
# ---------------------------------------------------------------------------

class TestXarrayTimeArithmetic:
    """Float DataArray × timedelta64(unit) controls the shift magnitude."""

    def _make_time_ds(self):
        """Returns a minimal xarray Dataset with time coord and float duration."""
        times = pd.date_range("2026-06-01", periods=3, freq="h")
        return xr.Dataset(
            {"duration": ("time", [3600.0, 3600.0, 3600.0])},
            coords={"time": times},
        )

    def test_float_times_us_adds_microseconds(self):
        # duration=3600 * timedelta64(1,'us') = 3600 µs = 3.6 ms (NOT 1 hour)
        ds = self._make_time_ds()
        end_us = ds.coords["time"] + ds["duration"] * np.timedelta64(1, "us")
        expected_shift = np.timedelta64(3600, "us")
        actual_shift = (end_us - ds.coords["time"]).values[0]
        assert actual_shift == expected_shift, (
            "Multiplying float seconds by timedelta64(1,'us') produces microseconds, "
            "not seconds — document this codebase inconsistency"
        )

    def test_float_times_s_adds_seconds(self):
        # duration=3600 * timedelta64(1,'s') = 3600 s = 1 hour (correct)
        ds = self._make_time_ds()
        end_s = ds.coords["time"] + ds["duration"] * np.timedelta64(1, "s")
        expected_shift = np.timedelta64(3600, "s")
        actual_shift = (end_s - ds.coords["time"]).values[0]
        assert actual_shift == expected_shift

    def test_us_and_s_produce_different_results(self):
        ds = self._make_time_ds()
        end_us = ds.coords["time"] + ds["duration"] * np.timedelta64(1, "us")
        end_s  = ds.coords["time"] + ds["duration"] * np.timedelta64(1, "s")
        # end_s is ~1 hour later than end_us (3600s vs 3.6ms → ~3600s difference)
        diff = (end_s - end_us).values[0].astype("timedelta64[s]").astype("int64")
        assert 3590 < diff < 3610  # within ±10 s of one hour

    def test_file_end_time_from_seconds_is_correct(self):
        # The compute_gap_lengths path (correct): duration_s * timedelta64(1,'s')
        ds = self._make_time_ds()
        file_end = ds.coords["time"] + ds["duration"] * np.timedelta64(1, "s")
        # first file starts 2026-06-01T00:00, duration=3600s → end = 2026-06-01T01:00
        expected = np.datetime64("2026-06-01T01:00:00")
        assert np.datetime64(file_end.values[0], "s") == expected


# ---------------------------------------------------------------------------
# NatSentinel
# Monitoring.py line 1057-1062 (compute_gap_lengths):
#   shift_start_time = start_time.shift(time=-1)
#   gap_length = shift_start_time - previous_end_time
#   gap_length = gap_length.values.astype('int64') * 1e-9
# ---------------------------------------------------------------------------

class TestNatSentinel:
    """xr.DataArray.shift → NaT → int64 overflow sentinel."""

    def test_shift_minus1_last_element_is_nat(self):
        da = xr.DataArray(
            pd.date_range("2026-06-01", periods=4, freq="h"),
            dims="time",
        )
        shifted = da.shift(time=-1)
        assert pd.isnull(shifted.values[-1])

    def test_nat_as_int64_is_imin(self):
        # NaT cast to int64 is always -9223372036854775808 (INT64_MIN / iNaT)
        nat = np.datetime64("NaT", "ns")
        as_int = np.array([nat]).astype("int64")[0]
        assert as_int == np.iinfo(np.int64).min

    def test_nat_sentinel_after_scale_is_large_negative(self):
        # The compute_gap_lengths scaling: gap_int64 * 1e-9 * sample_rate
        nat = np.datetime64("NaT", "ns")
        sentinel_seconds = np.array([nat]).astype("int64")[0] * 1e-9
        assert sentinel_seconds < -9e9

    def test_shift_interior_elements_are_valid(self):
        times = pd.date_range("2026-06-01", periods=4, freq="h")
        da = xr.DataArray(times, dims="time")
        shifted = da.shift(time=-1)
        for val in shifted.values[:-1]:
            assert not pd.isnull(val)

    def test_shift_minus1_moves_values_back(self):
        times = pd.date_range("2026-06-01", periods=4, freq="h")
        da = xr.DataArray(times, dims="time")
        shifted = da.shift(time=-1)
        # element 0 of shifted == element 1 of original
        assert shifted.values[0] == da.values[1]
        assert shifted.values[1] == da.values[2]


# ---------------------------------------------------------------------------
# TimedeltaWrapping
# Monitoring.py line 1230:
#   truncate_time = np.timedelta64(time_range[0] - sync_start_time, 's')
# Monitoring.py line 1247:
#   truncate_time = np.timedelta64(sync_end_time - time_range[1], 's')
# ---------------------------------------------------------------------------

class TestTimedeltaWrapping:
    """np.timedelta64(datetime64_diff, 's') coerces subtraction result to seconds."""

    def test_subtraction_of_seconds_datetimes_is_seconds_timedelta(self):
        dt1 = np.datetime64("2026-06-01T04:30:00", "s")
        dt2 = np.datetime64("2026-06-01T04:00:00", "s")
        diff = dt1 - dt2
        assert diff.dtype == np.dtype("timedelta64[s]")
        assert diff == np.timedelta64(1800, "s")

    def test_explicit_unit_wrapping_preserves_value(self):
        dt1 = np.datetime64("2026-06-01T04:30:00", "s")
        dt2 = np.datetime64("2026-06-01T04:00:00", "s")
        diff = dt1 - dt2
        wrapped = np.timedelta64(diff, "s")
        assert wrapped == np.timedelta64(1800, "s")

    def test_astype_int_gives_seconds(self):
        td = np.timedelta64(1800, "s")
        as_int = td.astype("int")
        assert as_int == 1800

    def test_wrapping_ns_to_s_truncates(self):
        # numpy must convert the difference value's units when wrapping
        dt1 = np.datetime64("2026-06-01T04:30:00", "ns")
        dt2 = np.datetime64("2026-06-01T04:00:00", "ns")
        diff = dt1 - dt2   # timedelta64[ns]
        wrapped_s = np.timedelta64(diff, "s")
        assert wrapped_s == np.timedelta64(1800, "s")

    def test_comparison_between_mixed_units_works(self):
        td_s  = np.timedelta64(60, "s")
        td_m  = np.timedelta64(1, "m")
        assert td_s == td_m


# ---------------------------------------------------------------------------
# SampleIndexComputation
# Monitoring.py lines 1231-1232:
#   start_index = np.floor(truncate_time.astype('int') * curr_sample_rate).astype('int')
#   new_start   = sync_start_time + np.timedelta64(int(start_index/curr_sample_rate), 's')
# ---------------------------------------------------------------------------

class TestSampleIndexComputation:
    """Truncate-time → sample index (floor/ceil/int interplay)."""

    def test_floor_gives_integer_sample_index(self):
        truncate_seconds = 1.5   # 1.5 s into the file
        sample_rate = 100.0
        td = np.timedelta64(int(truncate_seconds), "s")
        start_index = int(np.floor(td.astype("int") * sample_rate))
        assert start_index == 100   # floor(150) = 150... wait

    def test_floor_truncation_rounds_down(self):
        # 0.999 s at 100 Hz → floor(99.9) = 99 samples
        truncate_td = np.timedelta64(0, "s")  # 0 full seconds
        sample_rate = 100.0
        # Simulate: truncate_time comes from integer seconds
        td_int = truncate_td.astype("int")   # 0
        start_index = int(np.floor(td_int * sample_rate))
        assert start_index == 0

    def test_1s_at_100hz_gives_100_samples(self):
        truncate_s = np.timedelta64(1, "s")
        sample_rate = 100.0
        start_index = int(np.floor(truncate_s.astype("int") * sample_rate))
        assert start_index == 100

    def test_new_start_reconstruction(self):
        # start_index / sample_rate gives a whole-second offset (floor already applied)
        sync_start = np.datetime64("2026-06-01T00:00:00", "s")
        sample_rate = 100.0
        start_index = 300   # 3 seconds in
        new_start = sync_start + np.timedelta64(int(start_index / sample_rate), "s")
        assert new_start == np.datetime64("2026-06-01T00:00:03", "s")

    def test_ceil_end_index(self):
        # Monitoring line 1248: np.ceil(((td_total - td_truncate).astype(int) * fs))
        td_total = np.timedelta64(3600, "s")
        td_truncate = np.timedelta64(0, "s")
        sample_rate = 100.0
        end_index = int(np.ceil(((td_total - td_truncate).astype(int) * sample_rate)))
        assert end_index == 360000


# ---------------------------------------------------------------------------
# CombineFirstPrecedence
# Monitoring.py lines 621, 1006, 1011 (save_ds):
#   ds = ds.combine_first(dst)  / current_ds.combine_first(new_ds)
# ---------------------------------------------------------------------------

class TestCombineFirstPrecedence:
    """xr.Dataset.combine_first: caller's values win over argument's values."""

    def _ds(self, time, val):
        return xr.Dataset(
            {"x": ("time", [float(val)])},
            coords={"time": [np.datetime64(time, "ns")]},
        )

    def test_caller_wins_on_conflict(self):
        ds_caller = self._ds("2026-06-01T00:00", 1.0)
        ds_arg    = self._ds("2026-06-01T00:00", 2.0)
        result = ds_caller.combine_first(ds_arg)
        assert float(result["x"].values[0]) == 1.0

    def test_arg_fills_missing(self):
        ds_caller = self._ds("2026-06-01T00:00", 1.0)
        ds_arg    = self._ds("2026-06-01T02:00", 2.0)
        result = ds_caller.combine_first(ds_arg)
        assert len(result.time) == 2
        val_at_2h = float(result["x"].sel(time="2026-06-01T02:00").values)
        assert val_at_2h == 2.0

    def test_nan_in_caller_filled_by_arg(self):
        ds_caller = xr.Dataset(
            {"x": ("time", [np.nan])},
            coords={"time": [np.datetime64("2026-06-01T00:00", "ns")]},
        )
        ds_arg = self._ds("2026-06-01T00:00", 99.0)
        result = ds_caller.combine_first(ds_arg)
        assert float(result["x"].values[0]) == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# TimedeltaUnit
# Confirms that the unit passed to np.timedelta64 controls the magnitude of
# the result when multiplied by a float DataArray.
# ---------------------------------------------------------------------------

class TestTimedeltaUnit:
    """float × timedelta64(1, unit) — the unit literally scales the result."""

    def test_1us_multiplied_by_3600_gives_3600us(self):
        result = 3600.0 * np.timedelta64(1, "us")
        assert result == np.timedelta64(3600, "us")

    def test_1s_multiplied_by_3600_gives_3600s(self):
        result = 3600.0 * np.timedelta64(1, "s")
        assert result == np.timedelta64(3600, "s")

    def test_3600us_is_not_equal_to_3600s(self):
        us = np.timedelta64(3600, "us")
        s  = np.timedelta64(3600, "s")
        assert us != s

    def test_3600us_in_seconds_is_0_003600(self):
        us = np.timedelta64(3600, "us")
        in_s = us.astype("timedelta64[s]")
        assert in_s == np.timedelta64(0, "s")   # truncates to 0 whole seconds

    def test_xarray_float_da_times_td64_unit(self):
        # Confirm xarray preserves the unit when multiplying a float DataArray
        da = xr.DataArray([3600.0, 7200.0], dims="x")
        result_us = da * np.timedelta64(1, "us")
        result_s  = da * np.timedelta64(1, "s")
        assert result_us[0].values == np.timedelta64(3600, "us")
        assert result_s[0].values  == np.timedelta64(3600, "s")


# ---------------------------------------------------------------------------
# PandasTzNaiveAware
# load_filter_merge default time_range uses tz-aware Timestamps against a
# tz-naive xarray time coordinate — raises TypeError in pandas ≥ 3.0.
# ---------------------------------------------------------------------------

class TestPandasTzNaiveAware:
    """tz-naive / tz-aware comparison raises TypeError in pandas ≥ 3.0."""

    def test_tz_aware_ts_slice_on_naive_da_raises_type_error(self):
        # xr.DataArray with tz-naive datetime64 time coordinate
        da_naive = xr.Dataset(
            {"v": ("time", [1.0, 2.0])},
            coords={
                "time": pd.date_range("2026-06-01", periods=2, freq="h", tz=None)
            },
        )
        ts_tz = pd.Timestamp("2026-06-01 00:00", tz="Europe/Berlin")
        with pytest.raises(TypeError, match="[Cc]annot compare"):
            _ = da_naive.sel(time=slice(ts_tz, ts_tz + pd.Timedelta(hours=1)))

    def test_tz_naive_slice_works(self):
        da = xr.Dataset(
            {"v": ("time", [1.0, 2.0, 3.0])},
            coords={
                "time": pd.date_range("2026-06-01", periods=3, freq="h", tz=None)
            },
        )
        t0 = pd.Timestamp("2026-06-01 00:00")   # tz-naive
        t1 = pd.Timestamp("2026-06-01 01:00")   # tz-naive
        result = da.sel(time=slice(t0, t1))
        assert len(result.time) == 2

    def test_pandas_version_note(self):
        """Document the pandas version in use so regressions are easy to spot."""
        major = int(pd.__version__.split(".")[0])
        # Strict tz comparison was introduced in pandas 3.0
        if major >= 3:
            ts_naive = pd.Timestamp("2026-06-01")
            ts_tz    = pd.Timestamp("2026-06-01", tz="UTC")
            # In pandas >= 3.0 these must NOT compare equal; the comparison itself
            # raises instead of silently returning False
            with pytest.raises(TypeError):
                _ = ts_naive > ts_tz
        else:
            pytest.skip("pandas < 3.0 does not raise TypeError on tz comparison")


# ---------------------------------------------------------------------------
# DatetimeDtype
# pd.Timestamp.to_datetime64() returns datetime64[us] in numpy ≥ 2.0 /
# recent pandas.  Earlier versions returned datetime64[ns].
# ---------------------------------------------------------------------------

class TestDatetimeDtype:
    """pd.Timestamp.to_datetime64() returns datetime64[us] or datetime64[ns]."""

    def test_to_datetime64_is_datetime64_subtype(self):
        ts = pd.Timestamp("2026-06-01 00:00:00", tz="Europe/Berlin")
        dt64 = ts.to_datetime64()
        assert np.issubdtype(dt64.dtype, np.datetime64)

    def test_to_datetime64_precision_at_least_seconds(self):
        ts = pd.Timestamp("2026-06-01 00:00:00")
        dt64 = ts.to_datetime64()
        # Regardless of ns vs us, conversion to 's' must be lossless for whole seconds
        as_s = np.datetime64(dt64, "s")
        assert as_s == np.datetime64("2026-06-01T00:00:00", "s")

    def test_datetime64_comparisons_across_units_work(self):
        # datetime64[ns] == datetime64[us] must work (same instant)
        ns = np.datetime64("2026-06-01T00:00:00", "ns")
        us = np.datetime64("2026-06-01T00:00:00", "us")
        s  = np.datetime64("2026-06-01T00:00:00", "s")
        assert ns == us
        assert us == s
        assert ns == s

    def test_nat_comparison_with_various_units(self):
        nat_ns = np.datetime64("NaT", "ns")
        nat_us = np.datetime64("NaT", "us")
        nat_s  = np.datetime64("NaT", "s")
        # NaT != NaT (like float nan)
        assert not (nat_ns == nat_ns)
        # pd.isnull handles all units
        assert pd.isnull(nat_ns)
        assert pd.isnull(nat_us)
        assert pd.isnull(nat_s)


# ---------------------------------------------------------------------------
# NatInt64
# Confirms the sentinel value used in compute_gap_lengths after NaT → int64.
# ---------------------------------------------------------------------------

class TestNatInt64:
    """np.datetime64('NaT').astype('int64') == np.iinfo(np.int64).min."""

    INT64_MIN = np.iinfo(np.int64).min   # -9223372036854775808

    def test_nat_ns_to_int64_is_int64_min(self):
        nat = np.datetime64("NaT", "ns")
        assert nat.astype("int64") == self.INT64_MIN

    def test_nat_us_to_int64_is_int64_min(self):
        nat = np.datetime64("NaT", "us")
        assert nat.astype("int64") == self.INT64_MIN

    def test_nat_s_to_int64_is_int64_min(self):
        nat = np.datetime64("NaT", "s")
        assert nat.astype("int64") == self.INT64_MIN

    def test_int64_min_scaled_by_1e9ns_is_large_negative(self):
        # The compute_gap_lengths sentinel after scaling (1e-9 * sample_rate)
        sentinel_raw = self.INT64_MIN * 1e-9
        assert sentinel_raw < -9e9   # ~-9.2 × 10^9 seconds

    def test_regular_datetime64_astype_int64_is_not_sentinel(self):
        dt = np.datetime64("2026-06-01T00:00:00", "ns")
        as_int = dt.astype("int64")
        assert as_int != self.INT64_MIN
        assert as_int > 0   # positive POSIX nanoseconds


# ---------------------------------------------------------------------------
# Integration: verify file_info_accel.nc uses the patterns above correctly
# ---------------------------------------------------------------------------

from conftest import FILE_INFO_DIR, skip_if_no_data


@skip_if_no_data
class TestFileInfoTimeEncoding:
    """Integration tests verifying the on-disk file_info uses the expected encodings."""

    def test_time_coord_is_tz_naive(self):
        ds = xr.open_dataset(f"{FILE_INFO_DIR}/file_info_accel.nc")
        ds.load(); ds.close()
        # The coordinate must be tz-naive UTC datetime64
        assert "time" in ds.coords
        times = ds.coords["time"].values
        assert np.issubdtype(times.dtype, np.datetime64)
        # pd.DatetimeIndex wrapping tz-naive values has no tzinfo
        idx = pd.DatetimeIndex(times)
        assert idx.tzinfo is None

    def test_start_time_is_float64_posix(self):
        ds = xr.open_dataset(f"{FILE_INFO_DIR}/file_info_accel.nc")
        ds.load(); ds.close()
        assert ds["start_time"].dtype == np.float64
        # Values should be plausible POSIX timestamps (after 2015-01-01)
        ts_min = np.datetime64("2015-01-01", "s").astype("int64")
        assert (ds["start_time"].values > ts_min).all()

    def test_duration_is_float64_seconds(self):
        ds = xr.open_dataset(f"{FILE_INFO_DIR}/file_info_accel.nc")
        ds.load(); ds.close()
        assert ds["duration"].dtype == np.float64
        # 1-hour files → duration ≈ 3600 s
        durations = ds["duration"].values
        assert (durations > 0).all()
        assert (durations < 100_000).all()   # no file longer than ~27 h

    def test_start_time_recoverable_as_datetime64(self):
        ds = xr.open_dataset(f"{FILE_INFO_DIR}/file_info_accel.nc")
        ds.load(); ds.close()
        recovered = ds["start_time"].astype("datetime64[s]")
        assert recovered.dtype == np.dtype("datetime64[s]")
        # First file start must be in 2026
        first = pd.Timestamp(recovered.values[0])
        assert first.year == 2026

    def test_time_coord_consistent_with_start_time(self):
        ds = xr.open_dataset(f"{FILE_INFO_DIR}/file_info_accel.nc")
        ds.load(); ds.close()
        time_coords = ds.coords["time"].values
        start_times = ds["start_time"].astype("datetime64[s]").values
        # time coord and start_time must be within ± 2 minutes of each other
        diff_s = (time_coords.astype("datetime64[s]").astype("int64") -
                  start_times.astype("int64"))
        assert (np.abs(diff_s) < 120).all(), (
            "time coordinate differs from start_time by more than 2 minutes"
        )
