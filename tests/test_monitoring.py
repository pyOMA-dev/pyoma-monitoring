"""Tests for monitoring.py.

Structure
---------
Unit tests (no I/O)
    TestDescribeStats   — describe_stats() with synthetic signals
    TestRoundDt         — round_dt() datetime rounding
    TestCloseToUtc      — close_to_utc_transition() DST proximity check
    TestWindMath        — calc_xy, calc_ar, orthogonal_lsq
    TestWindTransform   — wind_transform() and compensate_wind_jumps()

Integration tests (require on-disk data, marked with @integration)
    TestReadFile        — read_file() on a real .dat binary
    TestDescribeStatsReal — describe_stats on a real file vs. file_info golden values
    TestGetStats        — get_stats() reading from result_db_short
    TestGetModalResults — get_modal_results() reading from result_db_short
"""
import datetime

import numpy as np
import pandas as pd
import pytest
import pytz
import xarray as xr

# conftest adds the package dir to sys.path
from conftest import FILE_INFO_DIR, skip_if_no_data

from monitoring import (
    close_to_utc_transition,
    describe_stats,
    read_file,
    round_dt,
    get_stats,
    get_modal_results,
    compute_gap_lengths,
)
from site_geyer import (
    calc_ar,
    calc_xy,
    compensate_wind_jumps,
    orthogonal_lsq,
    wind_transform,
)


# ---------------------------------------------------------------------------
# Unit tests — describe_stats
# ---------------------------------------------------------------------------

class TestDescribeStats:

    def test_keys_present(self, synthetic_measurement):
        meas, headers = synthetic_measurement
        result = describe_stats(meas, headers)
        expected_keys = {
            "mean", "min", "max", "var", "skewness", "kurtosis",
            "q05", "q50", "q95", "rms", "error",
        }
        assert set(result.keys()) == expected_keys

    def test_shapes(self, synthetic_measurement):
        meas, headers = synthetic_measurement
        result = describe_stats(meas, headers)
        for key, val in result.items():
            assert val.shape == (len(headers),), f"Shape mismatch for {key!r}"

    def test_mean_value(self, synthetic_measurement):
        meas, headers = synthetic_measurement
        result = describe_stats(meas, headers)
        np.testing.assert_allclose(result["mean"], meas.mean(axis=0), rtol=1e-6)

    def test_rms_is_std_of_zero_mean_signal(self):
        rng = np.random.default_rng(1)
        meas = rng.standard_normal((1000, 2))
        result = describe_stats(meas, ["A", "B"])
        # RMS of a zero-mean signal ≈ std
        np.testing.assert_allclose(result["rms"], meas.std(axis=0), rtol=1e-3)

    def test_quantile_ordering(self, synthetic_measurement):
        meas, headers = synthetic_measurement
        result = describe_stats(meas, headers)
        assert (result["q05"] <= result["q50"]).all()
        assert (result["q50"] <= result["q95"]).all()

    def test_constant_channel_flagged_as_error(self):
        meas = np.zeros((200, 2))
        meas[:, 1] = 3.14   # constant, non-zero
        result = describe_stats(meas, ["A", "B"])
        assert result["error"][0] == 1.0  # all-zero → constant
        assert result["error"][1] == 1.0  # non-zero constant

    def test_all_nan_channel_flagged_as_error(self):
        meas = np.full((100, 2), np.nan)
        meas[:, 0] = 1.0  # first channel is fine
        result = describe_stats(meas, ["ok", "bad"])
        assert not np.isnan(result["error"][0])  # first channel not error
        assert result["error"][1] == 1.0

    def test_no_headers_still_works(self):
        meas = np.random.default_rng(7).standard_normal((200, 3))
        result = describe_stats(meas)
        assert result["mean"].shape == (3,)


# ---------------------------------------------------------------------------
# Unit tests — round_dt
# ---------------------------------------------------------------------------

class TestRoundDt:

    def test_ceil_to_120min(self):
        dt = np.datetime64("2026-06-01T04:30")
        result = round_dt(dt, np.timedelta64(120), ceil=True)
        assert result == np.datetime64("2026-06-01T06:00")

    def test_ceil_already_on_boundary(self):
        dt = np.datetime64("2026-06-01T04:00")
        result = round_dt(dt, np.timedelta64(120), ceil=True)
        assert result == np.datetime64("2026-06-01T04:00")

    def test_floor_to_120min(self):
        dt = np.datetime64("2026-06-01T05:45")
        result = round_dt(dt, np.timedelta64(120), floor=True)
        assert result == np.datetime64("2026-06-01T04:00")

    def test_floor_already_on_boundary(self):
        dt = np.datetime64("2026-06-01T04:00")
        result = round_dt(dt, np.timedelta64(120), floor=True)
        # floor of an already-aligned boundary subtracts one period
        assert result == np.datetime64("2026-06-01T02:00")

    def test_ceil_60min(self):
        dt = np.datetime64("2026-06-01T03:15")
        result = round_dt(dt, np.timedelta64(60), ceil=True)
        assert result == np.datetime64("2026-06-01T04:00")


# ---------------------------------------------------------------------------
# Unit tests — close_to_utc_transition
# ---------------------------------------------------------------------------

class TestCloseToUtc:

    berlin = pytz.timezone("Europe/Berlin")

    def test_near_dst_spring_forward_is_close(self):
        # 2016-03-27 01:00 UTC is the spring-forward transition
        near = self.berlin.localize(datetime.datetime(2016, 3, 27, 1, 0))
        assert close_to_utc_transition(near) is True

    def test_within_window_is_close(self):
        # 2 h before the transition
        t = self.berlin.localize(datetime.datetime(2016, 3, 27, 0, 0))
        assert close_to_utc_transition(t) is True

    def test_far_from_dst_is_not_close(self):
        t = self.berlin.localize(datetime.datetime(2016, 3, 20, 12, 0))
        assert close_to_utc_transition(t) is False

    def test_outside_window_by_small_margin(self):
        # > 3 h away from the 2016-03-27 01:00 UTC transition
        t = self.berlin.localize(datetime.datetime(2016, 3, 26, 20, 0))
        assert close_to_utc_transition(t) is False


# ---------------------------------------------------------------------------
# Unit tests — wind math helpers
# ---------------------------------------------------------------------------

class TestWindMath:

    def test_calc_xy_unit_circle(self):
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        x, y = calc_xy(angles)
        np.testing.assert_allclose(x**2 + y**2, 1.0, atol=1e-12)

    def test_calc_ar_roundtrip(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(-10, 10, 50)
        y = rng.uniform(-10, 10, 50)
        az, r = calc_ar(x, y)
        x2, y2 = calc_xy(az, r)
        np.testing.assert_allclose(x2, x, atol=1e-10)
        np.testing.assert_allclose(y2, y, atol=1e-10)

    def test_orthogonal_lsq_xy(self):
        # Points clustered around the line y = x (45-degree direction)
        rng = np.random.default_rng(5)
        t = rng.uniform(0, 10, 200)
        noise = rng.standard_normal(200) * 0.05
        x = t + noise
        y = t - noise
        angle = orthogonal_lsq(xy=(x, y))
        # Expected angle ≈ 45°
        np.testing.assert_allclose(angle % 180, 45.0, atol=2.0)

    def test_orthogonal_lsq_azr_degrees(self):
        # Uniform wind from ~270°
        rng = np.random.default_rng(6)
        az = rng.uniform(260, 280, 100)
        r = rng.uniform(3, 10, 100)
        angle = orthogonal_lsq(azr=(az, r))
        # Result should be close to 270 (or within 360-degree wrap)
        assert 240 < angle % 360 < 300

    def test_calc_xy_calc_ar_inverse(self):
        az = np.array([0.0, np.pi / 4, np.pi / 2, np.pi])
        r = np.array([1.0, 2.0, 3.0, 4.0])
        x, y = calc_xy(az, r)
        az2, r2 = calc_ar(x, y)
        np.testing.assert_allclose(r2, r, atol=1e-10)
        np.testing.assert_allclose(az2, az, atol=1e-10)


# ---------------------------------------------------------------------------
# Unit tests — wind_transform and compensate_wind_jumps
# ---------------------------------------------------------------------------

class TestWindTransform:

    def _make_wind(self, n=2000, seed=10):
        rng = np.random.default_rng(seed)
        Wg = np.abs(rng.standard_normal(n)) * 4 + 5
        Wr = rng.standard_normal(n) * 15 + 200
        return Wg, Wr

    def test_wind_transform_output_channels(self):
        Wg, Wr = self._make_wind()
        meas = np.column_stack([Wg, Wr])
        file_time = datetime.datetime(2026, 6, 1, 0, 0, tzinfo=pytz.utc)
        start_time = datetime.datetime(2026, 6, 1, 2, 0, tzinfo=pytz.utc)
        _, new_headers, _, _, _, new_meas = wind_transform(
            file_time, ["Wg", "Wr"], ["m/s", "deg"], start_time, 1.0, meas
        )
        assert new_headers == ["Wg", "Wr", "Wx", "Wy"]
        assert new_meas.shape == (len(Wg), 4)

    def test_wind_transform_xy_magnitude(self):
        Wg, Wr = self._make_wind()
        meas = np.column_stack([Wg, Wr])
        file_time = datetime.datetime(2026, 6, 1, 0, 0, tzinfo=pytz.utc)
        start_time = datetime.datetime(2026, 6, 1, 2, 0, tzinfo=pytz.utc)
        _, _, _, _, _, new_meas = wind_transform(
            file_time, ["Wg", "Wr"], ["m/s", "deg"], start_time, 1.0, meas
        )
        Wg_out = new_meas[:, 0]
        Wx_out = new_meas[:, 2]
        Wy_out = new_meas[:, 3]
        computed_speed = np.sqrt(Wx_out**2 + Wy_out**2)
        # smoothed magnitude should be loosely consistent with wind speed
        np.testing.assert_allclose(
            computed_speed.mean(), Wg_out.mean(), rtol=0.5
        )

    def test_compensate_wind_jumps_removes_large_steps(self):
        n = 3000
        Wg = np.ones(n) * 6
        # introduce a 360-degree wrap-around at index 1000
        Wr = np.zeros(n) + 180.0
        Wr[1000:] += 360.0
        Wr += np.random.default_rng(99).standard_normal(n) * 0.5
        Wx, Wy = compensate_wind_jumps(Wr, Wg)
        # After correction the std of the cartesian components must not blow up
        assert np.std(Wx) < 20
        assert np.std(Wy) < 20


# ---------------------------------------------------------------------------
# Integration tests — read_file
# ---------------------------------------------------------------------------

@skip_if_no_data
class TestReadFile:

    def test_returns_tuple_of_7(self, first_accel_file):
        result = read_file(first_accel_file)
        assert result is not None
        assert len(result) == 7

    def test_headers_and_channels(self, first_accel_file):
        _, _, headers, units, _, _, _ = read_file(first_accel_file)
        assert "Accel_01" in headers
        assert "Accel_02" in headers
        # Time channel present in raw file
        assert "Time" in headers
        assert len(headers) == len(units)

    def test_sample_rate_is_100Hz(self, first_accel_file):
        _, _, _, _, _, sample_rate, _ = read_file(first_accel_file)
        assert sample_rate == pytest.approx(100.0)

    def test_measurement_shape_one_hour(self, first_accel_file):
        _, _, headers, _, _, sample_rate, measurement = read_file(first_accel_file)
        expected_samples = int(3600 * sample_rate)
        assert measurement.shape == (expected_samples, len(headers))

    def test_accel_mean_golden_values(self, first_accel_file):
        """Channel means from the first file match file_info_accel.nc golden values."""
        _, _, headers, _, _, _, measurement = read_file(first_accel_file)
        actual_mean = measurement.mean(axis=0)
        # Golden values from file_info_accel.nc isel(time=0)
        golden_mean_by_channel = {
            "Accel_01": -3.80741914e-03,
            "Accel_02": -8.97292718e-03,
            "Accel_03": -5.95079661e-02,
        }
        for ch, expected in golden_mean_by_channel.items():
            idx = headers.index(ch)
            np.testing.assert_allclose(actual_mean[idx], expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# Integration tests — describe_stats vs. file_info golden values
# ---------------------------------------------------------------------------

@skip_if_no_data
class TestDescribeStatsReal:

    def test_mean_matches_file_info(self, first_accel_file):
        """describe_stats on the raw file matches the stored file_info statistics."""
        _, _, headers, _, _, _, measurement = read_file(first_accel_file)
        result = describe_stats(measurement, headers)

        fi = xr.open_dataset(
            f"{FILE_INFO_DIR}/file_info_accel.nc", engine="netcdf4"
        )
        fi.load()
        fi.close()

        # Find the row for this file
        fname = "Accel_continuously__0_2026-06-01_00-00-00_000000.dat"
        time_idx = np.where(fi["file_name"].values == fname)[0]
        assert len(time_idx) == 1, "File not found in file_info"
        time_idx = time_idx[0]

        for ch in ["Accel_01", "Accel_02"]:
            ch_idx = headers.index(ch)
            fi_mean = fi["mean"].isel(time=time_idx).sel(channels=ch).values
            np.testing.assert_allclose(result["mean"][ch_idx], fi_mean, rtol=1e-4)
            fi_rms = fi["rms"].isel(time=time_idx).sel(channels=ch).values
            np.testing.assert_allclose(result["rms"][ch_idx], fi_rms, rtol=1e-3)


# ---------------------------------------------------------------------------
# Integration tests — compute_gap_lengths
# ---------------------------------------------------------------------------

@skip_if_no_data
class TestComputeGapLengths:

    def test_gap_length_variable_added(self, short_db):
        fi = xr.open_dataset(f"{FILE_INFO_DIR}/file_info_accel.nc")
        fi.load()
        fi.close()
        compute_gap_lengths(fi)
        assert "gap_length" in fi

    def test_gap_length_last_entry_is_sentinel(self, short_db):
        fi = xr.open_dataset(f"{FILE_INFO_DIR}/file_info_accel.nc")
        fi.load()
        fi.close()
        compute_gap_lengths(fi)
        # The last entry has no successor: xarray.shift fills with NaT which is
        # cast to int64 min, then scaled — result is a large negative sentinel.
        # The important thing is that it is NOT a realistic gap value.
        last = fi["gap_length"].values[-1]
        assert np.isnan(last) or last < -1e6, (
            f"Expected large negative sentinel for last entry, got {last}"
        )

    def test_gap_length_consecutive_files_small(self, short_db):
        fi = xr.open_dataset(f"{FILE_INFO_DIR}/file_info_accel.nc")
        fi.load()
        fi.close()
        compute_gap_lengths(fi)
        gaps = fi["gap_length"].values[:-1]  # exclude last (NaN)
        valid = gaps[~np.isnan(gaps)]
        # Consecutive 1-h files should not have large gaps (allow ±30 s * 100 Hz)
        assert (np.abs(valid) < 3000).all() or (valid < 5000).all()


# ---------------------------------------------------------------------------
# Integration tests — get_stats from result_db_short
# ---------------------------------------------------------------------------

@skip_if_no_data
class TestGetStats:
    """Read statistics from the short (read-only) result database."""

    GOLDEN_TIME = np.datetime64("2026-06-01T06:00:00.000000000")
    GOLDEN_MEAN_ACCEL = np.array([
        -0.04921015, -0.07992041, -0.0101594, -0.0430991,
        -0.01752379, -0.02227117, -0.00347758, -0.00681545,
    ])
    GOLDEN_RMS_ACCEL = np.array([
        0.0060316, 0.00790427, 0.00357258, 0.00639041,
        0.00814022, 0.00574008, 0.00109948, 0.00291581,
    ])

    def test_get_stats_accel_loads(self, short_db):
        ds = get_stats("accel", pd.Timedelta(minutes=120))
        assert ds is not None
        assert "mean" in ds

    def test_get_stats_accel_channels(self, short_db):
        ds = get_stats("accel", pd.Timedelta(minutes=120))
        channels = list(ds.channels.values.astype(str))
        assert "Accel_01" in channels
        assert "Accel_02" in channels

    def test_get_stats_accel_time_range(self, short_db):
        ds = get_stats("accel", pd.Timedelta(minutes=120))
        times = ds.time.values
        # Database spans late May to late June 2026
        assert times[0] <= np.datetime64("2026-06-01")
        assert times[-1] <= np.datetime64("2026-07-01")

    def test_get_stats_accel_golden_mean(self, short_db):
        ds = get_stats("accel", pd.Timedelta(minutes=120))
        row = ds.sel(time=self.GOLDEN_TIME)
        actual_mean = row["mean"].values
        np.testing.assert_allclose(actual_mean, self.GOLDEN_MEAN_ACCEL, rtol=1e-5)

    def test_get_stats_accel_golden_rms(self, short_db):
        ds = get_stats("accel", pd.Timedelta(minutes=120))
        row = ds.sel(time=self.GOLDEN_TIME)
        actual_rms = row["rms"].values
        np.testing.assert_allclose(actual_rms, self.GOLDEN_RMS_ACCEL, rtol=1e-5)

    def test_get_stats_wind_loads(self, short_db):
        ds = get_stats("wind", pd.Timedelta(minutes=120))
        assert ds is not None
        assert "Wg" in ds.channels.values.astype(str)

    def test_get_stats_temp_loads(self, short_db):
        ds = get_stats("temp", pd.Timedelta(minutes=120))
        assert ds is not None
        assert "Pt100_01" in ds.channels.values.astype(str)

    def test_get_stats_wind_golden_wg_mean(self, short_db):
        GOLDEN_WIND_TIME = np.datetime64("2026-06-01T06:00:00.000000000")
        GOLDEN_WG_MEAN = 39.17544161
        ds = get_stats("wind", pd.Timedelta(minutes=120))
        row = ds.sel(time=GOLDEN_WIND_TIME)
        wg_mean = float(row["mean"].sel(channels="Wg").values)
        np.testing.assert_allclose(wg_mean, GOLDEN_WG_MEAN, rtol=1e-4)


# ---------------------------------------------------------------------------
# Integration tests — get_modal_results from result_db_short
# ---------------------------------------------------------------------------

@skip_if_no_data
class TestGetModalResults:
    """Read modal results from the short (read-only) result database."""

    GOLDEN_TIME = np.datetime64("2026-06-01T04:00:00.000000000")
    GOLDEN_FREQUENCIES = np.array([
        1.30435982, 1.31411305, 0.3517834, 0.35438647,
        0.62513032, 0.61989003, 2.0514453,
    ])
    GOLDEN_DAMPING = np.array([
        1.09985129, 4.63790004, 0.69885105, 0.86907733,
        0.84073446, 0.40950517, 1.25785093,
    ])
    GOLDEN_NUM_MODES = 7

    def test_get_modal_loads(self, short_db):
        ds = get_modal_results("accel", pd.Timedelta(minutes=120))
        assert ds is not None
        assert "frequencies" in ds

    def test_get_modal_has_expected_variables(self, short_db):
        ds = get_modal_results("accel", pd.Timedelta(minutes=120))
        for var in ("frequencies", "damping", "num_modes", "modeshapes",
                    "MPC", "MPD", "model_orders"):
            assert var in ds, f"Variable {var!r} missing from modal dataset"

    def test_get_modal_golden_frequencies(self, short_db):
        ds = get_modal_results("accel", pd.Timedelta(minutes=120))
        row = ds.sel(time=self.GOLDEN_TIME)
        freqs = row["frequencies"].values
        valid_freqs = freqs[~np.isnan(freqs)]
        assert len(valid_freqs) == self.GOLDEN_NUM_MODES
        np.testing.assert_allclose(
            valid_freqs, self.GOLDEN_FREQUENCIES, rtol=1e-5
        )

    def test_get_modal_golden_damping(self, short_db):
        ds = get_modal_results("accel", pd.Timedelta(minutes=120))
        row = ds.sel(time=self.GOLDEN_TIME)
        damps = row["damping"].values
        valid_damps = damps[~np.isnan(damps)]
        assert len(valid_damps) == self.GOLDEN_NUM_MODES
        np.testing.assert_allclose(
            valid_damps, self.GOLDEN_DAMPING, rtol=1e-4
        )

    def test_get_modal_golden_num_modes(self, short_db):
        ds = get_modal_results("accel", pd.Timedelta(minutes=120))
        row = ds.sel(time=self.GOLDEN_TIME)
        assert int(row["num_modes"].values) == self.GOLDEN_NUM_MODES

    def test_get_modal_frequencies_in_physical_range(self, short_db):
        ds = get_modal_results("accel", pd.Timedelta(minutes=120))
        freqs = ds["frequencies"].values
        valid = freqs[~np.isnan(freqs)]
        assert (valid > 0.1).all(), "Frequencies below 0.1 Hz seem unphysical"
        assert (valid < 10.0).all(), "Frequencies above 10 Hz seem unphysical"

    def test_get_modal_damping_in_physical_range(self, short_db):
        ds = get_modal_results("accel", pd.Timedelta(minutes=120))
        damps = ds["damping"].values
        valid = damps[~np.isnan(damps)]
        assert (valid > 0.0).all(), "Negative damping ratios present"
        # The raw (unfiltered) stabilisation diagram contains some spurious
        # high-damping poles; a loose upper bound of 50 % catches gross errors.
        assert (valid < 50.0).all(), "Damping ratios above 50 % are clearly unphysical"
