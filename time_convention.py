"""Single source of truth for timezone/format conversions in the monitoring pipeline.

All methods are vectorised: they accept and return arrays / Series / xarray objects.
No per-scalar wrappers are created.
"""
import datetime

import dateutil.rrule
import numpy as np
import pandas as pd
import pytz
import xarray as xr


class TimeConvention:
    """Centralised timezone policy and conversion helpers.

    Owns the project timezone and the canonical on-disk storage format.
    The storage format is:
      - ``time`` coordinate: tz-naive UTC ``datetime64[ns]``
      - ``start_time`` / ``file_time`` variables: float64 POSIX seconds
    """

    def __init__(self, tz: str = "Europe/Berlin"):
        self.tz = pytz.timezone(tz)

    def to_storage_coord(self, aware) -> np.datetime64:
        """Berlin-aware datetime → tz-naive UTC ``datetime64`` for DB storage.

        Replaces the inline pattern::

            pd.Timestamp(sync_time).tz_convert('UTC').tz_localize(None).to_datetime64()
        """
        return pd.Timestamp(aware).tz_convert("UTC").tz_localize(None).to_datetime64()

    def to_local(self, stored) -> pd.Timestamp:
        """Stored tz-naive coordinate → Berlin-aware Timestamp (for slicing/display).

        Applies the "double-localisation" pattern used in ``create_stats`` and
        ``create_modal_results``: the stored UTC-naive value is treated as a
        Berlin-naive value and localised as Berlin.  Times inside the spring-forward
        DST gap map to ``NaT``.

        Replaces::

            pd.Timestamp(stored).tz_localize('Europe/Berlin', nonexistent='NaT')
        """
        return pd.Timestamp(stored).tz_localize(str(self.tz), nonexistent="NaT")

    def posix_to_datetime64(self, posix_s):
        """Float POSIX seconds → ``datetime64[s]`` (vectorised).

        Replaces ``da.astype('datetime64[s]')`` call sites.
        """
        if isinstance(posix_s, xr.DataArray):
            return posix_s.astype("datetime64[s]")
        return np.asarray(posix_s).astype("datetime64[s]")

    def to_posix(self, dt) -> float:
        """Tz-aware datetime → float POSIX seconds.

        Replaces ``.timestamp()`` calls for tz-aware ``datetime``/``pd.Timestamp``.
        """
        return dt.timestamp()

    def make_index(self, dtstart, until, minutes: int):
        """Build the rrule time index used in ``create_stats``/``create_modal_results``.

        Args:
            dtstart: start of the iteration (Berlin-naive or Berlin-aware datetime).
            until:   end of the iteration (inclusive, same timezone convention).
            minutes: step size in minutes.

        Returns:
            ``(aware_list, naive_array)`` where ``aware_list`` is a list of
            Berlin-aware ``pd.Timestamp`` objects and ``naive_array`` is a
            numpy array of tz-naive UTC ``datetime64[ns]`` values.

        Replaces the inline rrule + list-comprehension block in ``create_stats``.
        """
        dtstart_dt = pd.Timestamp(dtstart).to_pydatetime()
        rule = dateutil.rrule.rrule(
            dateutil.rrule.MINUTELY,
            interval=minutes,
            dtstart=dtstart_dt,
            until=until,
            cache=True,
        )
        aware_list = [pd.Timestamp(ts, tz=str(self.tz)) for ts in rule]
        naive_array = np.array(
            [ts.tz_convert("UTC").tz_localize(None) for ts in aware_list],
            dtype="datetime64[ns]",
        )
        return aware_list, naive_array

    def is_near_dst_transition(self, t, hours: int = 3) -> bool:
        """Return True if *t* falls within ±hours of any DST transition.

        Replaces ``close_to_utc_transition()`` in ``monitoring.py``.

        Args:
            t:     A timezone-aware or timezone-naive datetime.  The comparison
                   is performed by stripping tzinfo (matching the original behaviour
                   which compared against naive UTC transition times).
            hours: Proximity window in hours (default 3).
        """
        t_naive = t.replace(tzinfo=None)
        delta = datetime.timedelta(hours=hours)
        for utc_transition in self.tz._utc_transition_times[100:106]:
            if utc_transition - delta < t_naive < utc_transition + delta:
                return True
        return False

    def gap_lengths(self, start_times, durations, sample_rates) -> np.ndarray:
        """Compute gap lengths in samples between consecutive files.

        Fixes Bug 2: the last element of ``start_times.shift(time=-1)`` is NaT;
        converting NaT directly to int64 gives INT64_MIN (≈ −9.2×10⁹), which then
        propagates as a very large negative number of samples.  This method masks
        that sentinel to ``np.nan`` before multiplication.

        Uses ``np.timedelta64(1, 's')`` (not ``'us'``) so that float-second
        durations add the correct number of seconds (fixes Bug 1 for gap
        arithmetic).

        Args:
            start_times:  xr.DataArray of float64 POSIX start times (seconds).
            durations:    xr.DataArray of float64 durations in **seconds**.
            sample_rates: xr.DataArray of float64 sample rates in Hz.

        Returns:
            numpy ndarray of gap lengths in samples; last element is ``np.nan``.
        """
        start_dt = start_times.astype("datetime64[s]")
        previous_end = start_dt + durations * np.timedelta64(1, "s")
        shift_start = start_dt.shift(time=-1)

        int64_min = np.iinfo(np.int64).min
        shift_s = shift_start.values.astype("int64")
        end_s = previous_end.values.astype("int64")

        gap_s = (shift_s - end_s).astype(float)
        gap_s[shift_s == int64_min] = np.nan

        return gap_s * sample_rates.values


# Module-level singleton using the project timezone.
TC = TimeConvention()
