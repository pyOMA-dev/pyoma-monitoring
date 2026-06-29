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

    The storage format uses two encodings for the same instant:

    * ``time`` coordinate — tz-naive UTC ``datetime64[ns]`` (xarray index).
    * ``start_time`` / ``file_time`` variables — float64 POSIX seconds.

    Attributes
    ----------
    tz : pytz.BaseTzInfo
        The project timezone, e.g. ``pytz.timezone('Europe/Berlin')``.
    """

    def __init__(self, tz: str = "Europe/Berlin"):
        """Initialise the timezone policy.

        Parameters
        ----------
        tz : str, optional
            IANA timezone name.  Defaults to ``'Europe/Berlin'`` (CET/CEST),
            the timezone of the monitored structure this pipeline was built for.
        """
        self.tz = pytz.timezone(tz)

    def to_storage_coord(self, aware) -> np.datetime64:
        """Convert a timezone-aware datetime to a tz-naive UTC ``datetime64``.

        This is the canonical conversion used before writing a time coordinate
        to the NetCDF database.  It is equivalent to::

            pd.Timestamp(aware).tz_convert('UTC').tz_localize(None).to_datetime64()

        Parameters
        ----------
        aware : datetime.datetime or pd.Timestamp
            A timezone-aware datetime.  ``pytz``-aware and ``dateutil``-aware
            objects are both accepted.

        Returns
        -------
        numpy.datetime64
            Tz-naive UTC ``datetime64`` suitable for use as an xarray
            ``time`` coordinate.
        """
        return pd.Timestamp(aware).tz_convert("UTC").tz_localize(None).to_datetime64()

    def to_local(self, stored) -> pd.Timestamp:
        """Convert a stored tz-naive coordinate back to a Berlin-aware Timestamp.

        Used in ``create_stats`` and ``create_modal_results`` to recover a
        Berlin-aware ``pd.Timestamp`` from the UTC-naive value that is stored
        in the database.

        .. note::

            This applies the "double-localisation" pattern: the stored UTC-naive
            value is *re-localised* as Berlin (not converted from UTC).  Times
            that fall inside the spring-forward DST gap (02:00–02:59 on the last
            Sunday in March in ``Europe/Berlin``) map to ``NaT`` rather than
            raising, because ``nonexistent='NaT'`` is passed to ``tz_localize``.

        Parameters
        ----------
        stored : numpy.datetime64, pd.Timestamp, or str
            A tz-naive datetime value as stored in the database ``time``
            coordinate.

        Returns
        -------
        pd.Timestamp
            Berlin-aware timestamp, or ``NaT`` if *stored* falls inside a DST
            spring-forward gap.
        """
        return pd.Timestamp(stored).tz_localize(str(self.tz), nonexistent="NaT")

    def posix_to_datetime64(self, posix_s):
        """Convert float POSIX seconds to ``datetime64[s]`` (vectorised).

        Convenience wrapper around ``.astype('datetime64[s]')`` that accepts
        both plain NumPy arrays and xarray ``DataArray`` objects.

        Parameters
        ----------
        posix_s : float, array-like, or xarray.DataArray
            POSIX timestamp(s) in seconds since the Unix epoch (1970-01-01 UTC).

        Returns
        -------
        numpy.ndarray or xarray.DataArray
            ``datetime64[s]`` representation of the same instant(s).
        """
        if isinstance(posix_s, xr.DataArray):
            return posix_s.astype("datetime64[s]")
        return np.asarray(posix_s).astype("datetime64[s]")

    def to_posix(self, dt) -> float:
        """Convert a timezone-aware datetime to float POSIX seconds.

        Parameters
        ----------
        dt : datetime.datetime or pd.Timestamp
            A timezone-aware datetime.

        Returns
        -------
        float
            Seconds since the Unix epoch (1970-01-01 00:00:00 UTC), including
            fractional seconds.
        """
        return dt.timestamp()

    def make_index(self, dtstart, until, minutes: int):
        """Build the regular time index used by ``create_stats`` and ``create_modal_results``.

        Generates a ``dateutil.rrule`` sequence with step *minutes* from
        *dtstart* to *until* (inclusive) and returns it in two parallel forms:
        a list of Berlin-aware ``pd.Timestamp`` objects (for local-time
        display and slice naming) and a NumPy array of tz-naive UTC
        ``datetime64[ns]`` values (for xarray database lookup).

        Parameters
        ----------
        dtstart : datetime-like
            Start of the iteration range (Berlin-naive or Berlin-aware).
        until : datetime-like
            End of the iteration range, inclusive (same timezone convention
            as *dtstart*).
        minutes : int
            Step size in minutes (e.g. 30, 60, 120).

        Returns
        -------
        aware_list : list of pd.Timestamp
            Berlin-aware timestamps at every step from *dtstart* to *until*.
        naive_array : numpy.ndarray of datetime64[ns]
            Corresponding tz-naive UTC ``datetime64[ns]`` values, suitable for
            set-difference operations against the database ``time`` coordinate.
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
        """Return ``True`` if *t* is within ±*hours* of any DST transition.

        Used in ``monitoring.close_to_utc_transition`` to skip files recorded
        during the ±3-hour window around clock changes, where device timestamps
        are unreliable.

        The comparison strips ``tzinfo`` from *t* before comparing against the
        naive UTC transition times stored in ``pytz``, matching the original
        behaviour of the inline check this method replaces.

        Parameters
        ----------
        t : datetime.datetime
            A datetime object (timezone-aware or timezone-naive).
        hours : int, optional
            Half-width of the exclusion window in hours.  Defaults to 3.

        Returns
        -------
        bool
            ``True`` if *t* falls within *hours* of any DST transition in the
            project timezone; ``False`` otherwise.
        """
        t_naive = t.replace(tzinfo=None)
        delta = datetime.timedelta(hours=hours)
        for utc_transition in self.tz._utc_transition_times[100:106]:
            if utc_transition - delta < t_naive < utc_transition + delta:
                return True
        return False

    def gap_lengths(self, start_times, durations, sample_rates) -> np.ndarray:
        """Compute gap lengths in samples between consecutive files.

        Calculates, for each file *i*, how many samples are missing between the
        end of file *i* and the start of file *i*+1.  The last element is
        always ``np.nan`` because there is no next file.

        Two correctness issues from the original implementation are fixed here:

        * **Bug 1 (unit)** — the duration column (float seconds) is multiplied
          by ``np.timedelta64(1, 's')`` rather than ``'us'``, so 3600 s adds
          one hour, not 3.6 ms.
        * **Bug 2 (NaT sentinel)** — ``.shift(time=-1)`` fills the last element
          with ``NaT``; casting ``NaT`` directly to ``int64`` yields
          ``INT64_MIN`` (≈ −9.2×10¹⁸), which would propagate as an enormous
          negative gap.  Elements equal to ``INT64_MIN`` are masked to
          ``np.nan`` before scaling.

        Parameters
        ----------
        start_times : xarray.DataArray
            Float64 POSIX start times in **seconds** (the ``start_time``
            variable from the file-info database).
        durations : xarray.DataArray
            Float64 recording durations in **seconds**.
        sample_rates : xarray.DataArray
            Float64 sample rates in Hz.

        Returns
        -------
        numpy.ndarray
            Gap lengths in **samples**.  The last element is ``np.nan``.
            Negative values indicate file overlap; zero means perfectly
            consecutive.
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
