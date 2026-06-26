"""Geyer mast monitoring — site-specific callbacks and registration.

All knowledge that is particular to the Geyer site lives here:
  * File-list discovery (Geyer-specific filename patterns)
  * Wind-direction math (wind_transform, compensate_wind_jumps, …)
  * FBG strain conversion and temperature compensation
  * Channel-to-DOF mapping for modal analysis
  * Synchronisation-era policy (the 2016/2018/2019 cutoffs)
  * Modal frequency bands for split_modepairs

Import this module to register and activate the Geyer site in the engine::

    import site_geyer          # registers on import
"""
# coding: utf-8

import contextlib
import datetime
import glob
import logging
import os
import re

import numpy as np
import pandas as pd
import pytz
import scipy.interpolate

import config
import fbg_strain_reader

logger = logging.getLogger(__name__)

berlin_dst = pytz.timezone('Europe/Berlin')

# ---------------------------------------------------------------------------
# Wind-direction math
# ---------------------------------------------------------------------------

def calc_xy(az, r=1):
    x = r * np.cos(az)
    y = r * np.sin(az)
    return x, y


def calc_ar(x, y):
    xy = x ** 2 + y ** 2
    r = np.sqrt(xy)
    az = np.arctan2(y, x)
    return az, r


def orthogonal_lsq(azr=None, xy=None, rad=False):
    assert azr is not None or xy is not None

    if azr is not None:
        assert len(azr) == 2
        az, r = azr
        if not rad:
            az = np.radians(az)
        x, y = calc_xy(az, r)
    else:
        x, y = xy
        az, r = calc_ar(x, y)

    vector = np.array([x, y]).T
    _U, _S, V_T = np.linalg.svd(vector, full_matrices=False)

    angle = np.arctan2(-V_T[1, 0], V_T[1, 1])

    x_, _ = calc_xy((az - angle), r)
    if np.sum(x_ < 0) > len(x_) / 2:
        angle += np.pi
    if not rad:
        angle = np.degrees(angle)
    return angle


def compensate_wind_jumps(Wr, Wg):
    """Compensate for averaged jumps of wind direction between 0° and 360°."""

    def running_mean(y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    Wr = np.copy(Wr)

    angle = orthogonal_lsq(azr=[Wr, Wg])

    Wr -= angle
    Wr[Wr > 180] -= 360
    Wr[Wr < -180] += 360
    d_Wr = Wr[1:] - Wr[:-1]

    d_allow = 30

    intp_inds = np.logical_or(d_Wr < -d_allow, d_Wr > d_allow)
    intp_inds = np.hstack((intp_inds, np.array([0])))
    intp_inds = np.array(intp_inds, dtype=bool)

    eps = 0.00001
    std = 0
    new_std = np.std(d_Wr)
    counter = 0
    interpol_length = 1000
    overlap = 0.25

    while True:
        if new_std == 0:
            break
        if np.abs((new_std - std) / (new_std + std) / 2) < eps:
            break

        counter += 1

        if counter > 15:
            break

        step = int(interpol_length * (1 - overlap))
        for i in range(0, len(Wr), step):
            imin, imax = (i - step, i + step)
            if imin < 0:
                imin = 0
            if imax > len(Wr):
                imax = len(Wr)

            this_intp_inds = intp_inds[imin:imax]
            this_t = np.linspace(0, len(this_intp_inds) - 1, len(this_intp_inds))

            this_Wr = np.copy(Wr[imin:imax])

            ql, qml, qmu, qu = np.percentile(this_Wr, [10, 40, 60, 90])

            if np.abs(qu - ql) > 60:
                this_intp_inds = np.logical_or(
                    np.logical_or(this_Wr > qu, this_Wr < ql), this_intp_inds
                )
            this_intp_inds = np.logical_and(
                np.logical_not(np.logical_and(this_Wr < qmu, this_Wr > qml)),
                this_intp_inds,
            )

            knot_inds = np.logical_not(this_intp_inds)
            interp_y = this_Wr[knot_inds]
            interp_x = this_t[knot_inds]

            interp_x_new = this_t[this_intp_inds]

            if not len(interp_x):
                continue
            if len(interp_x_new) / len(interp_x) > 1:
                continue

            _k = min(len(interp_x), 5)
            spl = scipy.interpolate.InterpolatedUnivariateSpline(
                interp_x, interp_y, k=5, ext=0, check_finite=False
            )

            interp_y_new = spl(interp_x_new)

            this_Wr[this_intp_inds] = interp_y_new

            Wr[imin:imax] = this_Wr

        Wr += angle
        angle = orthogonal_lsq(azr=[Wr, Wg])

        Wr -= angle
        Wr[Wr > 180] -= 360
        Wr[Wr < -180] += 360

        d_Wr = np.diff(Wr)

        intp_inds = np.logical_or(d_Wr < -d_allow, d_Wr > d_allow)
        intp_inds = np.hstack((intp_inds, np.array([0])))
        intp_inds = np.array(intp_inds, dtype=bool)

        std = new_std
        new_std = np.std(d_Wr)

    Wr += angle

    x, y = calc_xy(az=np.radians(Wr), r=Wg)

    x = running_mean(x, 10)
    y = running_mean(y, 10)

    return x, y


def wind_transform(file_time, headers, units, start_time, sample_rate, measurement):
    """Convert raw Wg/Wr channels to Wg, Wr, Wx, Wy."""
    new_headers = []
    new_meas = []
    new_units = []
    for pair in [('Wg', 'Wr'), ('Wg_top', 'Wr_top')]:
        inds = []
        for header in pair:
            try:
                index = headers.index(header)
            except ValueError:
                break
            if index is None:
                break
            inds.append(index)
        else:
            Wg = measurement[:, inds[0]]
            Wr = measurement[:, inds[1]]

            Wx, Wy = compensate_wind_jumps(Wr, Wg)

            Wr_, Wg = calc_ar(Wx, Wy)
            Wr_ = np.degrees(Wr_)
            Wr_[Wr_ < 0] += 360

            new_meas.append(Wg)
            new_meas.append(Wr_)
            new_meas.append(Wx)
            new_meas.append(Wy)

            new_headers.append(pair[0])
            new_headers.append(pair[1])
            if 'top' in pair[0]:
                new_headers.append('Wx_top')
                new_headers.append('Wy_top')
            else:
                new_headers.append('Wx')
                new_headers.append('Wy')

            new_units.append(units[inds[0]])
            new_units.append(units[inds[1]])
            new_units.append(units[inds[0]])
            new_units.append(units[inds[0]])

    measurement = np.vstack(new_meas).T

    return file_time, new_headers, new_units, start_time, sample_rate, measurement


# ---------------------------------------------------------------------------
# FBG strain conversion and temperature compensation
# ---------------------------------------------------------------------------

def strain_manipulate_transform(
    start_time, headers, units, end_time, sample_rate, measurement, quantity, temp_slice=None
):
    """Convert FBG wavelengths to strain and temperature with optional temp compensation."""
    measurement, _deltas = fbg_strain_reader.manipulate_data(
        measurement, start_time, sample_rate,
        previous_a=None, previous_delta=None, previous_start_time=None,
    )

    S1 = 6.37E-06
    S2 = 7.46E-09
    k = 0.772
    alpha_steel = 12e-6
    alpha_glass = 0.55e-6

    strain_t = config.strain_t.get(quantity)
    if strain_t is None:
        raise RuntimeError(f"No strain_t mapping for quantity: {quantity}")

    new_measurement = np.zeros_like(measurement)

    for ind, header in enumerate(headers):
        if header in config.temp_channels:
            T = (
                -S1 / 2 / S2
                + np.sqrt(
                    (S1 ** 2 + 4 * S2 * np.log(measurement[:, ind] / config.initial_wl[header]))
                    / 4 / S2 ** 2
                )
                + 22.5
            )
            new_measurement[:, ind] = T

    if temp_slice is not None:
        _start_t, hds_t, _, _dur_t, _sample_rate_temp, meas_temp = temp_slice
        if quantity == 'strain_rosettes':
            temp_comp = {
                'A_Temp'  : 0.5 * meas_temp[:, hds_t.index('Pt100_01')] + 0.5 * meas_temp[:, hds_t.index('Pt100_04')],
                'B_Temp'  : meas_temp[:, hds_t.index('Pt100_04')],
                'C_Temp'  : 0.25 * meas_temp[:, hds_t.index('Pt100_04')] + 0.75 * meas_temp[:, hds_t.index('Pt100_03')],
                'D_Temp_2': 0.33 * meas_temp[:, hds_t.index('Pt100_03')] + 0.67 * meas_temp[:, hds_t.index('Pt100_02')],
            }
        elif quantity == 'strain_bolts':
            temp_comp = {
                '8_Temp' : 0.54 * meas_temp[:, hds_t.index("Pt100_01")] + 0.46 * meas_temp[:, hds_t.index("Pt100_04")],
                '9_Temp' : 0.63 * meas_temp[:, hds_t.index("Pt100_01")] + 0.37 ** meas_temp[:, hds_t.index("Pt100_04")],
                '10_Temp': 0.72 * meas_temp[:, hds_t.index("Pt100_01")] + 0.28 * meas_temp[:, hds_t.index("Pt100_04")],
            }
        else:
            temp_comp = {}
    else:
        temp_comp = {}
        for ind, header in enumerate(headers):
            if header in config.temp_channels:
                temp_comp[header] = new_measurement[:, ind]

    for channel, header in enumerate(headers):
        if header in config.temp_channels:
            pass
        else:
            comp_chan = strain_t[header]
            t = temp_comp[comp_chan]
            strain = (
                1 / k * (
                    np.log(measurement[:, channel] / config.initial_wl[header])
                    - S1 * (t - 22.5)
                    - S2 * (t - 22.5) ** 2
                )
                - (alpha_steel - alpha_glass) * (t - 22.5)
            )
            new_measurement[:, channel] = strain
            units[channel] = 'm/m'

    mean_ = np.nanmean(new_measurement, axis=0)
    _new_measurement_2 = new_measurement - mean_

    return start_time, headers, units, end_time, sample_rate, new_measurement


# ---------------------------------------------------------------------------
# Site transform callbacks
# ---------------------------------------------------------------------------

def _geyer_wind_transform(start_time, headers, units, end_time, sample_rate, measurement,
                           quantity=None, **kwargs):
    return wind_transform(start_time, headers, units, end_time, sample_rate, measurement)


def _geyer_strain_transform(start_time, headers, units, end_time, sample_rate, measurement,
                              quantity=None, start_time_local=None, duration=None, **kwargs):
    """Geyer FBG strain transform — only applies to pre-2018 files."""
    if start_time_local is None or start_time_local >= pd.Timestamp('2018-01-12', tz='Europe/Berlin'):
        return start_time, headers, units, end_time, sample_rate, measurement

    import monitoring as _m  # late import — monitoring already loaded by this point

    file_info_temp = kwargs.get('file_info_temp')

    temp_slice = None
    if file_info_temp is not None and duration is not None:
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            temp_slice = _m.get_slice(
                start_time_local, duration, quantity='temp',
                file_info=file_info_temp, upsample_fs=sample_rate,
            )

    if temp_slice is None:
        logger.warning(
            "Can't read measurement file for temperature compensation %s:%s. Skipping!",
            start_time_local, quantity,
        )
        return None

    Pt_ind = np.array(['Pt100' in head for head in temp_slice[1]])
    Pt_mean = np.nanmean(temp_slice[5][:, Pt_ind], axis=1)
    for ind, head in enumerate(temp_slice[1]):
        if 'Pt100' not in head:
            continue
        if np.nanmax(temp_slice[5][:, ind]) > 80 or np.nanmin(temp_slice[5][:, ind]) < -40:
            logger.info(
                'Channel %s out of bounds. Max: %s, Min: %s',
                head, np.nanmax(temp_slice[5][:, ind]), np.nanmin(temp_slice[5][:, ind]),
            )
            temp_slice[5][:, ind] = Pt_mean

    return strain_manipulate_transform(
        start_time, headers, units, end_time, sample_rate, measurement, quantity, temp_slice
    )


# ---------------------------------------------------------------------------
# Geometry / DOF setup for modal analysis
# ---------------------------------------------------------------------------

def _geyer_setup_accel(headers):
    ref_channels = [headers.index('Accel_01'), headers.index('Accel_02')]
    accel_channels = list(range(len(headers)))
    disp_channels = []
    chan_dofs_dict = {
        'Accel_01':     [1, 90,  0],
        'Accel_02':     [1, 180, 0],
        'Accel_03':     [4, 180, 0],
        'Accel_04':     [4, 90,  0],
        'Accel_05':     [5, 180, 0],
        'Accel_06':     [5, 90,  0],
        'Accel_07':     [6, 180, 0],
        'Accel_08':     [6, 90,  0],
        'Accel_01_top': [3, 270, 0],
        'Accel_02_top': [3, 180, 0],
        'Accel_03_top': [2, 270, 0],
        'Accel_04_top': [2, 0,   0],
    }
    return ref_channels, accel_channels, disp_channels, chan_dofs_dict


def _geyer_setup_strain_rosettes(headers):
    ref_channels = [
        headers.index('A_zt'),
        headers.index('B_zt'),
        headers.index('C_zt'),
        headers.index('D_zt'),
    ]
    accel_channels = []
    disp_channels = list(range(len(headers)))
    chan_dofs_dict = {
        'A_z' : ['A_z_1',  0, 90],
        'A_t' : ['A_t_1',  0, 0 ],
        'A_zt': ['A_zt_1', 0, 45],
        'B_z' : ['B_z_1',  0, 90],
        'B_t' : ['B_t_1',  0, 0 ],
        'B_zt': ['B_zt_1', 0, 45],
        'C_z' : ['C_z_1',  0, 90],
        'C_t' : ['C_t_1',  0, 0 ],
        'C_zt': ['C_zt_1', 0, 45],
        'D_z' : ['D_z_1',  0, 90],
        'D_t' : ['D_t_1',  0, 0 ],
        'D_zt': ['D_zt_1', 0, 45],
    }
    return ref_channels, accel_channels, disp_channels, chan_dofs_dict


# ---------------------------------------------------------------------------
# Synchronisation-era policy
# ---------------------------------------------------------------------------

def _geyer_sync_policy(start_time, file_time, duration):
    """Return the correct synchronised start time for the Geyer mast recording eras."""
    if start_time < berlin_dst.localize(datetime.datetime(2016, 12, 15)):
        return start_time
    if start_time < berlin_dst.localize(datetime.datetime(2018, 1, 25)):
        return file_time - duration
    if start_time < berlin_dst.localize(datetime.datetime(2019, 4, 30)):
        return start_time
    return start_time


# ---------------------------------------------------------------------------
# Site data constants
# ---------------------------------------------------------------------------

_GEYER_MODAL_BANDS = [
    (0.34, 0.38),
    (0.60, 0.65),
    (1.2,  1.4),
    (2.0,  2.15),
    (3.2,  3.55),
]

_GEYER_ERROR_RULES = {
    'accel':           {'kurtosis_max': 5, 'kurtosis_min': -2},
    'strain_rosettes': {'kurtosis_max': 5, 'kurtosis_min': -2},
}


# ---------------------------------------------------------------------------
# File-list discovery
# ---------------------------------------------------------------------------

def _geyer_get_file_list(origin, reduced=False, file_info=None):
    """Return all data file paths for the given Geyer origin."""
    path = os.path.join(config.file_root_path, config.subpaths[origin])
    path = os.path.normpath(path)
    if not os.path.exists(path):
        raise RuntimeError(f"{path} does not exist.")

    if origin == 'accel':
        file_list = glob.glob(os.path.join(path, 'Accel_continuously__*'))
        if not reduced:
            for pat in [
                'Alle_3h_00_00__1_2015-04-*', 'Alle_3h_00_00__1_2015-05-0*',
                'Alle_3h_00_00__3_2015-04-*', 'Alle_3h_00_00__3_2015-05-0*',
                'Alle_3h_03_00__2_2015-04-*', 'Alle_3h_03_00__2_2015-05-0*',
                'Alle_3h_03_00__4_2015-04-*', 'Alle_3h_03_00__4_2015-05-0*',
                'Alle_3h_06_00__3_2015-04-*', 'Alle_3h_06_00__3_2015-05-0*',
                'Alle_3h_06_00__5_2015-04-*', 'Alle_3h_06_00__5_2015-05-0*',
                'Alle_3h_09_00__4_2015-04-*', 'Alle_3h_09_00__4_2015-05-0*',
                'Alle_3h_09_00__6_2015-04-*', 'Alle_3h_09_00__6_2015-05-0*',
                'Alle_3h_12_00__5_2015-04-*', 'Alle_3h_12_00__5_2015-05-0*',
                'Alle_3h_12_00__7_2015-04-*', 'Alle_3h_12_00__7_2015-05-0*',
                'Alle_3h_15_00__6_2015-04-*', 'Alle_3h_15_00__6_2015-05-0*',
                'Alle_3h_15_00__8_2015-04-*', 'Alle_3h_15_00__8_2015-05-0*',
                'Alle_3h_18_00__7_2015-04-*', 'Alle_3h_18_00__7_2015-05-0*',
                'Alle_3h_18_00__9_2015-04-*', 'Alle_3h_18_00__9_2015-05-0*',
                'Alle_3h_21_00__8_2015-04-*', 'Alle_3h_21_00__8_2015-05-0*',
                'Alle_3h_21_00__10_2015-04-*', 'Alle_3h_21_00__10_2015-05-0*',
                'Wind_kontinuierlich__*',
            ]:
                file_list += glob.glob(os.path.join(path, pat))

    elif origin == 'wind':
        file_list = glob.glob(os.path.join(path, 'Wind_continuously__*'))
        if not reduced:
            file_list += glob.glob(os.path.join(path, 'Wind_kontinuierlich__*'))

    elif origin == 'temp':
        file_list = glob.glob(os.path.join(path, 'Temp_continuously__*'))
        if not reduced:
            file_list += glob.glob(os.path.join(path, 'Temp_konti_*'))

    elif origin == 'strain':
        file_list = []
        paths = [path]
        if not reduced:
            paths.append(os.path.join(path, 'binary_files_unusable'))
        for path_2 in paths:
            for filename in os.listdir(path_2):
                file, ext = os.path.splitext(filename)
                if ext == '.npz':
                    continue
                if ext == '.bz2':
                    file, ext = os.path.splitext(file)
                if file.endswith('spec'):
                    continue
                if ext == '.txt':
                    file = re.sub(r"strain-[1-4]", "", file)
                    if not file.endswith('_'):
                        continue
                    if os.path.join(path_2, file) in file_list:
                        continue
                    file_list.append(os.path.join(path_2, file))
                elif ext == '.bin':
                    file_list.append(os.path.join(path_2, filename))
    else:
        logger.warning('origin was neither accel nor wind nor strain nor temp. filelist is empty')
        file_list = []

    if reduced and file_info is not None:
        filename_list = [os.path.basename(f) for f in file_list]
        dset = set(filename_list).difference(file_info['file_name'].variable.data)
        logger.debug('%d %s', len(dset), dset)
        file_list = [os.path.join(path, filename) for filename in dset]

    return file_list


# ---------------------------------------------------------------------------
# Wind-direction circular mean for describe_stats
# ---------------------------------------------------------------------------

def _geyer_channel_mean_fn(header, measurement, headers):
    """Return orthogonal-LSQ mean for wind-direction channels; None for all others."""
    suffix = header.replace('Wr', '')
    wx_key = 'Wx' + suffix
    wy_key = 'Wy' + suffix
    if wx_key in headers and wy_key in headers:
        Wx = measurement[:, headers.index(wx_key)]
        Wy = measurement[:, headers.index(wy_key)]
        return orthogonal_lsq(xy=(Wx, Wy))
    return None


# ---------------------------------------------------------------------------
# Pre-processing channel selection for OMA
# ---------------------------------------------------------------------------

_GEYER_PREPROC_CHANNELS = {
    'strain_rosettes': [
        'A_z', 'A_t', 'A_zt',
        'B_z', 'B_t', 'B_zt',
        'C_z', 'C_t', 'C_zt',
        'D_z', 'D_t', 'D_zt',
    ],
    'strain_bolts': [
        '10_z1', '10_z2',
        '8_z1', '8_z2', '8_z3',
        '9_z1', '9_z2', '9_z3',
    ],
}


# ---------------------------------------------------------------------------
# Site registration
# ---------------------------------------------------------------------------

def register_geyer_site():
    """Register and activate the Geyer site in the monitoring engine.

    Called automatically when this module is imported. Can be called again
    safely (idempotent — it simply overwrites the registry entry).
    """
    import monitoring as _m  # late import avoids circular dependency at load time

    geyer_site = _m.Site(
        name="geyer",
        transforms={
            'wind':            _geyer_wind_transform,
            'strain_rosettes': _geyer_strain_transform,
            'strain_bolts':    _geyer_strain_transform,
        },
        setup_prep={
            'accel':           _geyer_setup_accel,
            'strain_rosettes': _geyer_setup_strain_rosettes,
        },
        error_rules=_GEYER_ERROR_RULES,
        sync_policy=_geyer_sync_policy,
        modal_bands=_GEYER_MODAL_BANDS,
        file_list_fn=_geyer_get_file_list,
        channel_mean_fn=_geyer_channel_mean_fn,
        preproc_channels=_GEYER_PREPROC_CHANNELS,
    )

    _m.register_site(geyer_site)
    _m.set_active_site(geyer_site)


register_geyer_site()
