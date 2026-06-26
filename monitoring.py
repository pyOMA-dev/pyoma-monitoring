'''
A data-organization, -storage and analysis scheme for long-term tower monitoring

store binary files on hard disk
time synchronization must be achieved prior to analysis:
  cases
   pc/controller is ahead of controller/pc
   daylight saving
   clock of pc/controller changed
   available information: 
     timestamp controller,  → Reference, because it does not change often and there is no daylight saving, though it may not relate to the “real time”
     timestamp pc, 
     filetime controller, 
     filetime pc
correct (and synchronize) files and store as binary on hard disk
store results in a database structure and link to the file on disk; preferred data-organization: 
using different averages 1m, 10m and 1 hour should be possible
    → create multiple databases

_____________________________________________________________________
Database: File Information (file_info_<quantity>.nc)

Coordinates: time (synchronized to controller time but in local timezone (w/ daylight saving), channels
Variables:

File Information:
file_name ('time',)
file_size ('time',)
file_time ('time',)
start_time ('time',)

Content Information:
num_channels ('time',)
units ('channels',)
sample_rate ('time',)
duration ('time',) (in seconds)
length ('time',) (in samples)
error ('time', 'channels')

Statistical Description of the recorded signal:
mean ('time', 'channels')
min ('time', 'channels')
max ('time', 'channels')
var ('time', 'channels')
skewness ('time', 'channels')
kurtosis ('time', 'channels')
q05 ('time', 'channels')
q50 ('time', 'channels')
q95 ('time', 'channels')
rms ('time', 'channels')

_____________________________________________________________________
Database: Statistics (stats_<quantity>.nc)

Coordinates: time, channels
Variables:

Content Information:
num_channels ('time',)
sample_rate ('time',)
length ('time',)
error ('time', 'channels')

Statistical Description of the sliced and processed signal:
mean ('time', 'channels')
min ('time', 'channels')
max ('time', 'channels')
var ('time', 'channels')
skewness ('time', 'channels')
kurtosis ('time', 'channels')
q05 ('time', 'channels')
q50 ('time', 'channels')
q95 ('time', 'channels')
rms ('time', 'channels')


_____________________________________________________________________
Database: Modal results (modal_<quantity>.nc)

Coordinates: time, channels, modes
Variables:

Content information of the sliced signal:
num_channels ('time',)
sample_rate ('time',)
length ('time',)
error ('time', 'channels')

Operational Modal Analysis
num_modes ('time',)
frequencies ('time', 'modes')
damping ('time', 'modes')
model_orders ('time', 'modes')
std_damping ('time', 'modes')
std_frequencies ('time', 'modes')
var_svd_psd ('time', 'channels')
modeshapes ('time', 'modes', 'channels')
MPC ('time', 'modes')
MPD ('time', 'modes')

Statistical description of the singular value PSD:
max_svd_psd ('time', 'channels')
mean_svd_psd ('time', 'channels')
min_svd_psd ('time', 'channels')
skewness_svd_psd ('time', 'channels')
kurtosis_svd_psd ('time', 'channels')
q05_svd_psd ('time', 'channels')
q50_svd_psd ('time', 'channels')
q95_svd_psd ('time', 'channels')
rms_svd_psd ('time', 'channels')
energy_svd_psd ('time', 'channels')

Statistical Description of the sliced and processed signal:
mean ('time', 'channels')
min ('time', 'channels')
max ('time', 'channels')
var ('time', 'channels')
skewness ('time', 'channels')
kurtosis ('time', 'channels')
q05 ('time', 'channels')
q50 ('time', 'channels')
q95 ('time', 'channels')
rms ('time', 'channels')
_____________________________________________________________________
'''
# coding: utf-8

import os
import glob
import sys
import contextlib
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import re
import pytz
import datetime
import time
import tzlocal
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

berlin_dst = pytz.timezone('Europe/Berlin')  # cet/cest

from time_convention import TC

# ---------------------------------------------------------------------------
# Site registry — must be defined before any site module is imported
# ---------------------------------------------------------------------------

@dataclass
class Site:
    """Contract between the generic engine and a site-specific implementation.

    All callables are pure functions with no side-effects on the engine state.
    Path / channel / range fields carry site-specific configuration that was
    previously read directly from config.py by the engine.
    """
    name: str
    # --- processing callbacks ---
    transforms: Dict[str, Callable]      # quantity -> callable(*Slice, quantity, **kw) -> Slice | None
    setup_prep: Dict[str, Callable]      # quantity -> callable(headers) -> (ref_ch, accel_ch, disp_ch, chan_dofs)
    error_rules: Dict[str, dict]         # quantity -> {kurtosis_max, kurtosis_min}
    sync_policy: Callable                # (start_time, file_time, duration) -> datetime
    modal_bands: List[tuple]             # [(f_lo, f_hi), …] for split_modepairs
    file_list_fn: Callable               # (origin, reduced, file_info) -> list[str]
    channel_mean_fn: Optional[Callable]  # (header, measurement, headers) -> float | None
    preproc_channels: Dict[str, list]    # quantity -> [channel names for OMA pre-processing]
    # --- site configuration (formerly in config.py) ---
    db_root_path: str
    slice_root_path: str
    modal_conf_dir: str
    file_root_path: str
    origins: Dict[str, str]              # quantity -> origin tag
    subpaths: Dict[str, str]             # origin tag -> relative filesystem path
    all_channels: Dict[str, list]        # quantity -> required channel list
    optional_channels: Dict[str, list]   # quantity -> optional channel list
    dtstarts: Dict[str, object]          # origin -> earliest datetime
    ranges: Dict[str, tuple]             # channel name -> (min, max) plausibility range

_SITES: Dict[str, Site] = {}
_active_site: Optional[Site] = None

# ---------------------------------------------------------------------------
# Mutable process-level state (not site-specific)
# ---------------------------------------------------------------------------
from collections import deque as _deque
_file_cache: _deque = _deque(maxlen=25)
_ds_cache: dict = {}
_pid: str = str(os.getpid())


def register_site(site: Site) -> None:
    _SITES[site.name] = site


def get_site(name: str) -> Site:
    return _SITES[name]


def set_active_site(site: Site) -> None:
    global _active_site
    _active_site = site


def get_active_site() -> Optional[Site]:
    return _active_site

import math
import numpy as np
import scipy.stats
import scipy.signal
import scipy.signal.ltisys
import scipy.interpolate

import bz2
import fbg_strain_reader
import gantner_reader


import pandas as pd
import xarray as xr

from MultiLock import MultiLock

from pyOMA.core.PreProcessingTools import GeometryProcessor,PreProcessSignals
from pyOMA.core.StabilDiagram import StabilCluster, StabilPlot
try:
    from pyOMA.GUI.StabilGUI import start_stabil_gui
except ImportError:
    start_stabil_gui = None
# from GUI.PlotMSHGUI import start_msh_gui
from pyOMA.core.VarSSIRef import VarSSIRef
#from PLSCF import PLSCF
#from SSIData import SSIDataMC

reader_tz = tzlocal.get_localzone()
        
        
def get_file_list(origin, reduced=False, file_info=None):
    """Return a list of data file paths for the given origin and site."""
    site = _active_site
    if site is None or site.file_list_fn is None:
        raise RuntimeError(
            "No file_list_fn registered. Import the appropriate site module first."
        )
    return site.file_list_fn(origin, reduced, file_info)

def read_file(path):
    '''
    read binary file from path
    return array, start_time, duration, headers, 
    path may be a csv,dat or bin file
    or a bzip2ed version of it
    or an incomplete path refering to the 8 files generated by illumisense (common path part of these files, ends in underscore, has not extension)
        illumisense files may be .txt or .txt.bz2
        
    we first try to read a .npz version for this file, if it exists
    then we try to open the bz2 file and pass the file descriptor to the following function
    then we  call the respective function to read the file
    
    the illumisense reader must handle bz2 internally
    
    Short comparison of slowdown and space saving using bz2. packed files
                time        speedup/slowdown [%]  filesize    space saving [%]
    .dat        1,64E-07    100,00                14400768    
    .dat.bz2    1,27E-06    12,91                 10300268    28,47
                    
    .csv        1,85E-07    100,00                40320184    
    .csv.bz2    9,55E-07    19,37                  9115468    77,39
                    
    .txt        8,32E-07    100,00              3260560549    
    .txt.bz2    2,29E-06    36,30                217251268    93,34
    .npz        3,92E-08    2125,37        
                    
    .bin        1,26E-04    100,00                34195960    
    .bin.bz2    1,26E-04    100,00                 2882224    91,57
    .npz        4,90E-07    25617,82        
    '''   
    if path.endswith('_'):
        assert (os.path.exists(path+'strain-1.txt') or os.path.exists(path+'strain-1.txt.bz2'))
    else:
        if not os.path.exists(path):
            logger.warning('File {} does not exist'.format(os.path.basename(path)))
            return None
    
    logger.info(f'Reading File: {os.path.basename(path)}')
    
    for f_dict in _file_cache:
        if f_dict.get('path', '') == path:
            logger.debug('Returning File from cache ')
            file_time = f_dict['file_time']
            file_size = f_dict['file_size']
            headers = f_dict['headers']
            units = f_dict['units']
            start_time = f_dict['start_time']
            sample_rate = f_dict['sample_rate']
            measurement = f_dict['measurement']

            return file_time, file_size, headers, units, start_time, sample_rate, measurement
    
    if path.endswith('_'):
        if os.path.exists(path+'strain-1.txt'):
            file_time = datetime.datetime.fromtimestamp(os.path.getmtime(path+'strain-1.txt'), tz=reader_tz)
        elif os.path.exists(path+'strain-1.txt.bz2'):
            file_time = datetime.datetime.fromtimestamp(os.path.getmtime(path+'strain-1.txt.bz2'), tz=reader_tz)
    else:
        file_time = datetime.datetime.fromtimestamp(os.path.getmtime(path), tz=reader_tz)
        
    if path.endswith('_'):
        #strain txt files, one for each channel of the interrogator
        file_size=0
        for j in [2,3,4]:
            if os.path.exists(path+'strain-{}.txt'.format(j)):
                file_size+=os.path.getsize(path+'strain-{}.txt'.format(j))
            if os.path.exists(path+'strain-{}.txt.bz2'.format(j)):
                file_size+=os.path.getsize(path+'strain-{}.txt.bz2'.format(j))
    else:
        file_size = os.path.getsize(path)
        
    file,ext = os.path.splitext(path)

    if ext == '.bz2':
        zipfile  = bz2.BZ2File(path, 'r')
        try:
            zipfile.seek(0,2)
            zipfile.seek(0,0)
        except (EOFError, OSError):
            logger.warning('BZ2File is corrupted: {}'.format(path))
            return None
        
        file,ext = os.path.splitext(file)
    else:
        zipfile = None
        
    try:
        in_dict = np.load(file+ext+'.npz', allow_pickle=True)
        headers = list(in_dict['headers'])
        units = list(in_dict['units'])
        start_time=in_dict['startTimestamp'].item()
        if isinstance(start_time, pd.Timestamp): start_time = start_time.to_pydatetime()# if from txt:  <class 'datetime.datetime'> <class 'pandas._libs.tslibs.timestamps.Timestamp'>
        if (ext == '.bin' or ext == '') and not start_time.tzinfo: start_time = fbg_strain_reader.localize(start_time)
        sample_rate=in_dict['sample_rate'].item()
        measurement=in_dict['measurement']
        
        _file_cache.append({'path': path,
                           'file_time':file_time,
                           'file_size':file_size,
                           'headers':headers,
                           'units':units,
                           'start_time':start_time,
                           'sample_rate':sample_rate,
                           'measurement':measurement
                           })
        
        return file_time, file_size, headers, units, start_time, sample_rate, measurement
        
    except Exception as e:
        if os.path.exists(file+'.npz'):
            logger.exception(e)
            os.remove(file+'.npz')
            
    if zipfile is not None:
        file = zipfile
    else:
        file=path
    try:
        if ext == '.dat':
            file_contents  = gantner_reader.read_bin(file) # <class 'datetime.datetime'> <class 'datetime.datetime'
        elif ext == '.csv':
            file_contents = gantner_reader.read_csv(file, path)
        elif ext == '.bin':
            file_contents = fbg_strain_reader.read_bin(file, path=path)
        elif ext == '':
            file_contents = fbg_strain_reader.read_strain_txt(file)
        elif ext==".filepart":
            return None
        else:
            logger.warning('Extension was neither .dat, .csv, .bin, or None but {}'.format(ext))
            return None
        
    except OSError as e:
        logger.exception(e)
        return None
        
    if file_contents is None:
        return None
    
    headers, units, start_time, sample_rate, measurement = file_contents 
     
    _file_cache.append({'path': path,
                       'file_time':file_time,
                       'file_size':file_size,
                       'headers':headers,
                       'units':units,
                       'start_time':start_time,
                       'sample_rate':sample_rate,
                       'measurement':measurement
                       })
    
    return file_time, file_size, headers, units, start_time, sample_rate, measurement
    

def get_file_info(origin: str, create_new: bool=False, **kwargs):
    site = _active_site
    ds_path = os.path.join(site.db_root_path, f'file_info_{origin}.nc')
    if not os.path.exists(ds_path):
        ds = xr.Dataset()
        ds.to_netcdf(ds_path, format='NETCDF4')
        ds.close()
        logger.warning(f"DB Path {ds_path} was recreated.")
        return None

    if not create_new:
        stat_result = os.stat(ds_path, follow_symlinks=True)
        cache_key = f'{origin}_file_info'
        cached = _ds_cache.get(cache_key)
        if cached is not None and cached['mtime'] == stat_result.st_mtime:
            logger.info('Getting file info for {} (cached)'.format(origin))
            ds = cached['ds']
        else:
            logger.info('Getting file info for {}'.format(origin))
            ds = xr.open_dataset(ds_path)
            ds.load()
            ds.close()
            _ds_cache[cache_key] = {'ds': ds, 'mtime': stat_result.st_mtime}
    else:
        ds = create_file_info(origin, **kwargs)
    compute_gap_lengths(ds)
    return ds     
   
def close_to_utc_transition(file_time, close_hours=3):
    result = TC.is_near_dst_transition(file_time, hours=close_hours)
    if result:
        logger.info('Timestamp is within +- {} hours of daylight saving transition time: {}'.format(close_hours, file_time))
    return result

def round_dt(dt: np.datetime64, duration: np.timedelta64,
             ceil: bool=True, floor: bool=False):
    
    if floor: ceil=False
    
    td = np.timedelta64(duration,'m')
    mindt = np.array(0,dtype='datetime64[ns]')
    mindt = np.datetime64('2015-05-20 00:00')
    #    convert dt to timedelta, compute remainder and add it 
    dt = (mindt - dt)            % td               + dt
    if floor: dt -= td
    
    return dt
    
def get_synchronized_time(start_time, file_time, duration):
    """Return the synchronised start time via the active site's sync policy."""
    site = _active_site
    if site is not None:
        return site.sync_policy(start_time, file_time, duration)
    return start_time


    
def create_file_info(origin: str, chunksize: int=50, skip_existing: bool=True, **kwargs):
    '''
    file_name, file_size, file_creation_time, num_channels, headers, units, sample_rate, type (qstation, labview), start_time,<- 1st run
    length (in samples), per channel: {errors, mean, min, max, var, skewness, kurtosis, q05, q95, q50, rms} <- 1st run
    next_file, gap_length (in_samples) <- 2nd run
    '''
    
    logger.info('Creating file info for {}'.format(origin))   
    ds_path = os.path.join(_active_site.db_root_path, 'file_info_{}.nc'.format(origin))
    
    if os.path.exists(ds_path):
        ds =  xr.open_dataset(ds_path)
        ds.load()
        ds.close()
    else:
        ds = xr.Dataset() 
        ds.to_netcdf(ds_path, format='NETCDF4')
        ds.close()
        logger.warn(f"DB Path {ds_path} was recreated.")
        
    
    reduced = kwargs.pop('reduced',True)
    filtered_list = kwargs.pop('filtered_list',False)
    
    if filtered_list and reduced and len(ds.dims)>0:
        filelist = get_file_list(origin, reduced, ds)
    else:
        filelist = get_file_list(origin, reduced)
        
    logger.info('Total number of files: {}'.format(len(filelist)))
    
    logger.info('Total number of files already read: {}'.format(len(ds.time) if 'time' in ds else 0))
        
    num_workers = kwargs.get('num_workers',1)
    this_worker = kwargs.get('this_worker',1)
    jobsize = int(np.ceil(len(filelist)/num_workers))
    start = int((this_worker-1)%num_workers)
    
    logger.debug('{} {}'.format(start, jobsize))
    i = 0
    
    while i*chunksize<jobsize:
        ds =  xr.open_dataset(ds_path)
        ds.load()
        
        changed = False
        for i, file in enumerate(filelist[start*jobsize:(start+1)*jobsize]):
        #for file in filelist[i * chunksize:(i+1) * chunksize]:
            now=time.time()
            file_name = os.path.basename(file)
            if 'file_name' in ds : 
                if (file_name == ds.get('file_name')).any():
                    if skip_existing:
                        logger.debug('{} already present in dataset. Skipping!'.format(file_name))
                        continue
                    else:
                        logger.info('{} already present in dataset. Updating!'.format(file_name))
                        ds = ds.where(file_name != ds.get('file_name'), drop=True)# drop the entry for this file_name
            
            file_contents = read_file(file)
                
            if file_contents is None:
                logger.warning('File unreadable, consider deleting it manually: {}'.format(file))
                continue
            
            file_time, file_size, headers, units, start_time, sample_rate,measurement = file_contents
            
            if close_to_utc_transition(file_time):
                continue
            
            duration = datetime.timedelta(seconds=measurement.shape[0]/sample_rate)
            
            dst = xr.Dataset()          
            
            dst['file_name'] = (('time'), np.array([file_name],dtype=str))
            dst['file_size'] = (('time'), [file_size])
            dst['file_time'] = (['time'], [TC.to_posix(file_time)])
            dst['start_time'] = (['time'], [TC.to_posix(start_time)])
            # convert back using dst['start_time'].astype('datetime64[s]')
            dst['num_channels'] = (['time'], [measurement.shape[1]])
            dst['units'] = (['channels'], np.array(units,dtype=str))
            dst['sample_rate'] = (['time'], [sample_rate])
            dst['duration']=(['time'],[np.asarray(duration, dtype='timedelta64[s]').astype(np.float64)])
            dst['duration'].attrs['units'] = 'seconds'
            dst['length']= (['time'], [measurement.shape[0]])
            
            this_dict=describe_stats(measurement, headers)
            
            for var_name, _value in this_dict.items():
                dst[var_name] = (['time','channels'], [this_dict[var_name]])
            
            sync_time = get_synchronized_time(start_time, file_time, duration)
    
            dst.coords['channels'] = np.array(headers,dtype=str)
            # TODO: time is not a unique identifier
            # if two files of the same quantity and origin were started at the same time (highly unlikely), results will get overwritten 
            dst.coords['time'] = [TC.to_storage_coord(sync_time)]

            ds=ds.combine_first(dst)
            changed=True
            logger.debug(f'Success: , {(file_name==ds["file_name"]).any().item()}, Length: , {len(ds.time)},:, {len(ds.channels)},  Loading Time: , {time.time()-now}, s')
        
        i += 1
        if changed:
            now=time.time()
            # logger.info('saving ')
            save_ds(ds, ds, ds_path, what='file_info', reload_current=True)
            # ds.to_netcdf(os.path.join(path,'result_db/file_info_{}.nc'.format(origin)), format='NETCDF4')  
            # logger.debug('{time.time()-now} s')
    ds.close()
    return ds


def describe_stats(measurement, headers=None, quantity=None):
    
    
    
    if headers is None:
        headers = ['' for channel in range(measurement.shape[1])]
    
    this_dict = {}         
    for key in ['mean','min','max','var','skewness','kurtosis','q05','q50','q95','rms','error']:
        this_dict[key]=np.array([np.nan for channel in headers])
         
    for channel, header in enumerate(headers):
        
        try:
            if np.isnan(measurement[:,channel]).all():
                this_dict['error'][channel]=True
                continue
            site = _active_site
            this_range = site.ranges.get(header, None) if site is not None else None

            site_mean_fn = site.channel_mean_fn if site is not None else None
            circular_mean = site_mean_fn(header, measurement, headers) if site_mean_fn is not None else None

            if circular_mean is not None:
                mean = circular_mean
                Wr = np.copy(measurement[:, channel])
                Wr -= mean
                Wr[Wr < -180] += 360
                Wr[Wr > 180] -= 360
                Wr += mean

                _nobs, (min_, max_), _, variance, skewness, kurtosis = scipy.stats.describe(Wr, nan_policy='omit')
                q05, q50, q95 = np.nanpercentile(Wr, q=[5, 50, 95])
                rms = np.sqrt(np.nanmean(np.square(Wr)))

            else:
                _nobs, (min_,max_), mean, variance, skewness, kurtosis = scipy.stats.describe(measurement[:,channel],nan_policy='omit')
                q05,q50,q95 = np.nanpercentile(measurement[:,channel], q=[5,50,95])
                rms = np.sqrt(np.nanmean(np.square(measurement[:,channel]-mean)))
                
            this_dict['mean'][channel]=mean
            this_dict['min'][channel]=min_
            this_dict['max'][channel]=max_
            this_dict['var'][channel]=variance
            this_dict['skewness'][channel]=skewness
            this_dict['kurtosis'][channel]=kurtosis
            this_dict['q05'][channel]=q05
            this_dict['q50'][channel]=q50
            this_dict['q95'][channel]=q95
            this_dict['rms'][channel]=rms
            this_dict['error'][channel]=False
            if min_ == max_:
               this_dict['error'][channel]=True
            if this_range is not None:
                if max_ > this_range[1]:
                    this_dict['error'][channel]=True
                if min_ < this_range[0]:
                    this_dict['error'][channel]=True
            if quantity is not None and _active_site is not None:
                rules = _active_site.error_rules.get(quantity, {})
                if rules.get('kurtosis_max') is not None and kurtosis > rules['kurtosis_max']:
                    this_dict['error'][channel] = True
                if rules.get('kurtosis_min') is not None and kurtosis < rules['kurtosis_min']:
                    this_dict['error'][channel] = True
                    
                    
                    
                    
            
        except Exception:
            raise
    
    return this_dict



def check_and_mark_errors(ds, new = True, check_kurtosis = False):
    '''
    checks and marks errors in the dataset ds
    new: discard existing error flags and only use newly created error flags
    
    returns: dataset ds with updated error dataarray
    '''
    
    for channel in ds.channels:
        this_range = _active_site.ranges.get(channel.variable.item(), None) if _active_site is not None else None
        this_range = None
                    
        error=ds.sel(channels=channel).get('error')
        if new:
            error=0
        
        
        error += ds.sel(channels=channel).get('min')==ds.sel(channels=channel).get('max')
        if this_range is not None:
            error += ds.sel(channels=channel).get('max')>this_range[1]
            error += ds.sel(channels=channel).get('min')<this_range[0]
        logger.debug(channel.channels.variable.data.item())
        if check_kurtosis:
            logger.debug(channel.channels.variable.data)
            error += ds.sel(channels=channel).get('kurtosis')>5
            error += ds.sel(channels=channel).get('kurtosis')<-2
        ds['error'].loc[:,channel] = error
    
    return ds
        
def get_stats(quantity: str, duration: pd.Timedelta, 
              file_info: xr.Dataset=None, create_new: bool=False, 
              **kwargs):
    
    site = _active_site
    minutes = int(duration.total_seconds()/60)
    ds_path = os.path.join(site.db_root_path, f'{minutes}-minutes/', 'stats_{}.nc'.format(quantity))

    if not os.path.exists(ds_path):
        logger.warning(f'Path for stats does not exist {ds_path}.')

    if file_info is None and create_new:
        raise RuntimeError('File info xarray has to be provided to create a statistical description')

    if not create_new:
        stat_result = os.stat(ds_path, follow_symlinks=True)
        cache_key = f'{quantity}_stats'
        cached = _ds_cache.get(cache_key)
        if cached is not None and cached['mtime'] == stat_result.st_mtime:
            logger.info('Getting statistics for {} (cached)'.format(quantity))
            ds = cached['ds']
        else:
            logger.info('Getting statistics for {}'.format(quantity))
            ds = xr.open_dataset(ds_path)
            ds.load()
            ds.close()
            _ds_cache[cache_key] = {'ds': ds, 'mtime': stat_result.st_mtime}
    else:
        ds = create_stats(quantity, duration, file_info, **kwargs)

    return ds
    
    
def create_stats(quantity: str, duration: pd.Timedelta, file_info: xr.Dataset, 
                 chunksize: int=10, skip_existing: bool=False, **kwargs):
    '''
    mean, min, max, var, skewness, kurtosis, q05, q95, q50, rms
    '''

    logger.info('Creating statistics for {} '.format(quantity))
    
    minutes = int(duration.total_seconds()/60)
    
    site = _active_site
    process_ds_path = os.path.join(site.db_root_path, f'{minutes}-minutes/', f'stats_{quantity}.{_pid}.nc')
    master_ds_path = os.path.join(site.db_root_path, f'{minutes}-minutes/', 'stats_{}.nc'.format(quantity))

    if os.path.exists(process_ds_path):
        process_ds = xr.open_dataset(process_ds_path)
        process_ds.load()
        process_ds.close()
    else:
        process_ds = xr.Dataset()
        os.makedirs(os.path.join(site.db_root_path, f'{minutes}-minutes/'), exist_ok=True)
        process_ds.to_netcdf(process_ds_path, format='NETCDF4')
        process_ds.close()

    if os.path.exists(master_ds_path):
        master_ds = xr.open_dataset(master_ds_path)
        master_ds.load()
        master_ds.close()
    else:
        master_ds = xr.Dataset()

    origin = site.origins[quantity]
    dtstart = site.dtstarts[origin]
    dtstart = kwargs.pop('dtstart', dtstart)
    dtstart = pd.Timestamp(dtstart).to_pydatetime()
    fi_time_max = (file_info.time + file_info.duration * np.timedelta64(1, 's')).max().values
    fi_time_max = round_dt(fi_time_max, duration, ceil=True)
    
    fi_time_max = pd.Timestamp(fi_time_max).to_pydatetime()
    
    until = kwargs.pop('until', fi_time_max)
    logger.debug(f'Analyzing time range: {dtstart} ... {until} <= {fi_time_max}')
    
    
    # time_iterator consists of non timezone-aware items,
    # each item will be converted to timezone-aware pd.Timestamp,
    # thus time_iterator is interpreted to be in "Europe/Berlin" with DST
    # conversion to UTC by ts.to_datetime64() results in:
    # even summer hours and uneven winter hours(for 120 minute slices)
    # the converted UTC times are the indices of the database
    # the only place, were timezone native times are used get_slice_corrected to generate the filename (=even hours year round)
    # file_info['start_time'/'file_time'] is using timestamps (which are UTC by definition = seconds since 1.1.1970 (UTC))
    # file_info['time'] is derived from tz-aware start_time and file_time in get_synchronized_time and converted to UTC

    _aware_iter, time_iter_naive = TC.make_index(dtstart, until, minutes)
    
    validate_slices = kwargs.pop('validate_slices', False)
    
    if skip_existing and not validate_slices and 'time' in master_ds.dims:
        # drop all time instances without results to try them again, in case previous runs failed before proceeding them
        stats_ds = master_ds.dropna(dim='time', how='all')
        # filter out errorneous files to not do them again every time
        stats_ds = stats_ds.time[~np.logical_or(stats_ds['error'].any(dim='channels'), stats_ds.length.isnull())]
        
        time_iterator = np.setdiff1d(time_iter_naive, stats_ds.time.data, assume_unique=True)
    else:
        time_iterator = time_iter_naive

    num_workers = kwargs.get('num_workers',1)
    this_worker = kwargs.get('this_worker',1)
    jobsize=int(np.ceil(len(time_iterator)/num_workers))
    start = int((this_worker-1)%num_workers)
    
    #logger.debug('this_worker: {}, num_workers: {}, duration: {}, quantity: {}'.format(this_worker, num_workers, duration, quantity))
    #return
    
    for i, time_ in enumerate(time_iterator[start*jobsize:(start+1)*jobsize]):
        logger.debug(time_)
        start_time = TC.to_local(time_)
        # if not (i+1)%50: print('.',end='', flush=True) 
        # if not (i+1)%2500: print('\n')
        
        if close_to_utc_transition(start_time):
            continue
        
        if 'time' in master_ds.coords:
            if (start_time.to_datetime64()==master_ds.coords['time']).any():
                if skip_existing:
                    # other checks may be added here
                    
                    if not validate_slices:
                        logger.debug(f'{str(start_time)} already present in master dataset. Skipping!')
                        continue
                    
                    if  master_ds.sel(time=start_time)['mean'].isnull().all():
                        try:
                            # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f): 
                            old_level = logger.getEffectiveLevel()
                            logger.setLevel(logging.WARNING)
                            data_slice = get_slice_corrected(start_time, duration, quantity, file_info, **kwargs)
                            logger.setLevel(old_level)
                        except Exception as e:
                            logger.exception(e)
                            data_slice=None
                        if data_slice is not None:
                            logger.info(f'{str(start_time)} empty in master dataset, but slice available. Updating!')
                        else:
                            logger.debug(f'{str(start_time)} empty but present in master dataset. Skipping!')
                            continue
                    else:
                        logger.debug(f'{str(start_time)} already present in master dataset. Skipping!')
                        continue
                else: 
                    logger.info(f'{str(start_time)} already present in master dataset. Updating!')
            
        if 'time' in process_ds.coords:
            if (start_time.to_datetime64()==process_ds.coords['time']).any():
                if skip_existing: 
                    # other checks may be added here
                    continue
                else:
                    logger.info(f'{str(start_time)} already present in process dataset. Updating!')
                    process_ds = process_ds.where((start_time.to_datetime64()!=process_ds.coords['time']), drop=True)
                    # other checks may be added here

        this_ds = xr.Dataset()
        
        try:
            old_level = logger.getEffectiveLevel()
            logger.setLevel(logging.WARNING)
            data_slice = get_slice_corrected(start_time, duration, quantity, file_info, **kwargs)
            logger.setLevel(old_level)
        except Exception as e:
            logger.exception(e)
            data_slice=None

        if data_slice is None:
            logger.debug('Returned Slice is empty. Skipping!')

        else:
            _actual_start_time, headers, _units, _end_time, sample_rate, measurement = data_slice

            this_ds['num_channels'] = (['time'], [measurement.shape[1]])#:len(headers),
            this_ds['sample_rate'] = (['time'], [sample_rate])#:sample_rate,
            this_ds['length']= (['time'], [measurement.shape[0]])
            this_dict = describe_stats(measurement, headers, quantity)

            for var_name, _value in this_dict.items():
                this_ds[var_name] = (['time','channels'], [this_dict[var_name]])

            this_ds.coords['channels'] = np.array(headers,dtype=str)

        # this_ds.coords['time'] = [np.asarray(start_time, dtype='datetime64[ns]')]
        this_ds.coords['time'] = [np.asarray(start_time.to_datetime64())]

        process_ds = process_ds.combine_first(this_ds)
        logger.debug('Success: {}, Length: {}'.format((start_time.to_datetime64()==process_ds.coords['time']).any().item(), len(process_ds.time)))
        
        if i>0 and not i%chunksize:
            # process_ds should not have changed on disk during processing
            process_ds = save_ds(process_ds, process_ds, process_ds_path, what='stats')

    # process_ds should not have changed on disk during processing
    process_ds = save_ds(process_ds, process_ds, process_ds_path, what='stats')

    # master_ds almost certainly has changed on disk if multiple workers were processing files
    master_ds = save_ds(process_ds, master_ds, master_ds_path, reload_current=True, what='stats')

    logger.debug(f'Removing temporary dataset at {process_ds_path}')
    os.remove(process_ds_path)

    return master_ds


def save_ds(new_ds, current_ds, savepath, what='modal', reload_current = False):
    '''
    defined behaviour:
        when creating new results:
            save to a unique netcdf file for each process in order to avoid loosing data
            unique netcdf files will be merged later
        when updating results:
            iterate over stats and existing modal results
            regenerate where necessary
            save to a unique netcdf file for each process
        upon process exit
            lock main netcdf
            merge main netcdf with processes netcdf while dropping conflicting indexes from main file

    '''
    logger.debug('')
    now=time.time()  
    
    with MultiLock(savepath):
        
        if reload_current and os.path.exists(savepath):
            if what=='modal':
                current_ds = xr.open_dataset(savepath, engine='h5netcdf')
            else:
                current_ds = xr.open_dataset(savepath)
            current_ds.load()
            current_ds.close()
            
        if not 'time' in new_ds:
            logger.debug('Not saving empty new dataset.')        
            return current_ds
        
        logger.info(f'Merging Datasets {os.path.split(savepath)[-1]}')
        if 'time' in current_ds:
            old_length=len(current_ds.time)
        else:
            old_length = 0
    
        if 'time' in current_ds:
    #         for start_time in new_ds.time:
    #             print('.',end='', flush=True)
    #             current_ds = current_ds.where((start_time!=current_ds.coords['time']), drop=True)
            dupes, _ind_new, _ind_current = np.intersect1d(new_ds.time, current_ds.time, assume_unique=True, return_indices=True)
            len_before = len(current_ds.time)
            current_ds = current_ds.drop_sel(time=dupes)
            logger.debug(f'dropped {len_before - len(current_ds.time)} results that were already present in the db.')
            current_ds = current_ds.combine_first(new_ds)
            # For datasets, ds0.combine_first(ds1) works similarly to xr.merge([ds0, ds1]),
            # except that xr.merge raises MergeError when there are conflicting values in
            # variables to be merged, whereas .combine_first defaults to the calling object's values.
            logger.info('Dataset length before/after: {}/{}, '.format(old_length, len(current_ds.time)))
    
        else:
            logger.warning("'Dataset' object has not attribute 'time'. Overwriting!")
            current_ds = new_ds
    
        logger.info(f'Saving Dataset to {savepath}')
        tempfile = savepath + '.tmp'
        if what=='modal':
            current_ds.to_netcdf(tempfile, engine='h5netcdf', invalid_netcdf=True)
        elif what=='stats':
            current_ds.to_netcdf(tempfile, format='NETCDF4')
        elif what=='file_info':
            current_ds.to_netcdf(tempfile, format='NETCDF4')
        
        if os.path.exists(savepath):
            os.remove(savepath)
        os.rename(tempfile, savepath)
        
        logger.debug('{} s'.format( now-time.time()))

    return current_ds


def compute_gap_lengths(file_info):
    '''
    check file_gaps (assumes files are sorted in time)
    this function only works reliably if all files have been read in before,
    therefore it can not be pre-computed and is re-computed every time the script is run

    in Peaks_*, when the file compression script starts, all data written afterwards to the currently active file was lost
    therefore we have a gap of some minutes every night around 1 AM CET/CEST
    other gaps may be due to pc restarts, controller restarts, ...

    Peaks_* files have to be interleaved sometime, when the recording stopped eg in channel 2 and continued in the next file in channel 3
    then there is a gap of (-1) sample which is removed during interleaving
    '''
    file_info['gap_length'] = xr.DataArray(
        TC.gap_lengths(file_info['start_time'], file_info['duration'], file_info['sample_rate']),
        dims=['time'],
    )

def get_slice(start_time, duration , quantity, file_info, upsample_fs=None):
    '''
    channels 'Tagessekunden' and 'Time' are always dropped
    '''
    
    time_range = (start_time.to_datetime64(), (start_time + duration).to_datetime64())
    
    site = _active_site
    channels_required = site.all_channels[quantity]
    origin = site.origins[quantity]
    _subpath = site.subpaths[origin]
#
    file_start_time = file_info.time
    #duration = file_info.duration
    #duration = ((file_info['length']+1)/file_info['sample_rate']).astype('timedelta64[s]') #duration.astype('float64')*(5/3)*1e-11 #duration in minutes
    file_end_time = file_info.time + file_info.duration * np.timedelta64(1, 's')
    
    # first select all files that end in time_range i.e 'time' is within time_range
    b1 = file_end_time>=np.datetime64(time_range[0])
    
    # then also select all files, that start in time_range i.e. 'time'+'duration' is within time_range
    b2 = file_start_time<=np.datetime64(time_range[1])
    
    # combine selectors and truncate dataset
    b = np.logical_and(b1, b2)
    file_info = file_info.where(b, drop=True)#.dropna(dim='time', how='all')
    
    if len(file_info['file_name']) == 0:
        logger.info('There is no file for: {} - {}; {} '.format(*time_range, quantity))
        return
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(file_info['file_name'])
        logger.debug(file_info['sample_rate'])
        logger.debug(file_info['gap_length'][:-1])
        
    logger.info('Extracting signal slice for {} starting at {} until {}'.format(quantity, *np.datetime_as_string(time_range, unit='m')))
    if len(file_info['file_name']) > 1 and (file_info['gap_length'][:-1]>0.).any():

        if quantity in [ 'temp', 'wind', 'strain_rosettes'] and (file_info['gap_length'][:-1]<32).all():
            #gaps of 32 samples are allowed and will be interpolated (in strains) in qstation records there should be bigger gaps or none
            pass
        else:
            logger.info('Files are not consecutive. Gap of max. length {} s between files.'.format((file_info['gap_length']/file_info['sample_rate'])[:-1].data.max()))
            return

    # logger.debug(file_info['error'])
    if 'Tagessekunden' in file_info.channels.data:
        # file_info = file_info.drop(['Tagessekunden'],'channels')
        file_info = file_info.drop_sel(channels=['Tagessekunden'])
    if 'Time' in file_info.channels.data:
        # file_info = file_info.drop(['Time'],'channels')
        file_info = file_info.drop_sel(channels=['Time'])
    # logger.debug(file_info)
    file_info = file_info.dropna(dim='channels', how='any')
    # logger.debug(file_info.channels)
    
    if channels_required is not None:
        channels_in_file = set(file_info.channels.data.astype(str))
        channels_to_drop = list(channels_in_file.difference(channels_required + site.optional_channels.get(quantity, [])))
        # file_info = file_info.drop(channels_to_drop,'channels')
        file_info = file_info.drop_sel(channels=channels_to_drop)
    # logger.debug(all_channels)  
    if not set(channels_required).issubset(channels_in_file):
        logger.warning('Not all channels present, missing {}. Skipping!'.format(set(channels_required).difference(channels_in_file)))
        return
    
    channels = list(file_info.dropna(dim='channels', how='any').channels.data.astype(str))
    logger.debug('Common Channels: {}'.format( channels))
    
    if len(file_info['file_name']) == 0:
        raise RuntimeError('How did this happen?')
   
    
    if (file_info['error']==1.).any():
        logger.debug('files are (partially) marked as errorneous. Affected channels: {}'.format(list(file_info.where(file_info['error'], drop=True).channels.data)))
        #return
    units = list(file_info['units'].isel(time=0).variable.data.astype(str))
    logger.debug(units)
    logger.debug(len(file_info['file_name']))
    
    # ensure time_range is fully covered by the selected files
    
    last_end = (file_info.time + file_info['duration'] * np.timedelta64(1, 's')).data[-1]
    first_start = file_info.time.data[0]
    
    if first_start > np.datetime64(time_range[0]) or last_end < np.datetime64(time_range[1]):
        logger.info('Time range {} - {} is not fully covered by files {}. Skipping!'.format(str(time_range[0]), str(time_range[1]), list(file_info.file_name.data)))
        return None
    
    
    all_measurements = []
    new_start = None
    new_end = None

    # loop over all files in the range, load them strip unneeded parts and concatenate them together
    for file_num in range(len(file_info.time)):
        file = file_info.isel(time=file_num)['file_name']
        
        gap_length = file_info.isel(time=file_num)['gap_length']
        _sample_rate = file_info.isel(time=file_num)['sample_rate']
        
        logger.debug(file.item())
        
        filename = file.item()
        if 'Peaks' in filename: filename= os.path.join('binary_files_unusable',filename)
        
        file_path = os.path.join(site.file_root_path, site.subpaths[origin], filename)
        
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f): 
            file_contents = read_file(file_path)
            
        if file_contents is None:
            logger.warning('File unreadable: {}'.format(file_path))
            return None
        curr_file_time, _curr_file_size, curr_headers, _curr_units, curr_start_time, curr_sample_rate, curr_measurement = file_contents
        
        start_index = 0
        end_index = curr_measurement.shape[0]
        
        # in case of strain measurements files may actually need to be interleaved
        # then one (or more) more line is needed from the end
        # start_time/end_time is the time instant between the first/last sample
        if  'strain' in quantity and time_range[0] < pd.Timestamp('2018-01-12', tz='Europe/Berlin').to_datetime64():
            startTimestamp, endTimestamp, firstChannel, lastChannel, firstIndex, lastIndex = fbg_strain_reader.read_bin(file_path, indices_only=True)
            num_full_lines = (lastIndex+1-firstIndex)/4
            assert num_full_lines == end_index-(firstChannel+(3-lastChannel))/4
            logger.debug('(lastIndex+1-firstIndex)/4 {}'.format( (lastIndex+1-firstIndex)/4))
            logger.debug('end_index-(firstChannel+(3-lastChannel))/4 {}'.format( end_index-(firstChannel+(3-lastChannel))/4))

            curr_duration = endTimestamp - startTimestamp - datetime.timedelta(seconds=1/(4*curr_sample_rate))
            curr_duration_ = num_full_lines/(curr_sample_rate)
            curr_sample_rate = num_full_lines/curr_duration.total_seconds()

            startTimestamp = startTimestamp - datetime.timedelta(seconds=firstChannel/(4*curr_sample_rate))
            endTimestamp = endTimestamp + datetime.timedelta(seconds=(3-lastChannel)/(4*curr_sample_rate))
            
            # in some cases labview stopped writing data to the file
            # in cases of early file ends, the end timestamp is computed from the linenumber and sample rate in fbg_strain_reader
            # however the file timestamp is still at the regular end of the file
            # so there is a large amount of data missing
            # in these cases we should not rely on the file time stamp
            sync_end_time = endTimestamp - datetime.timedelta(seconds=1/(4*curr_sample_rate))
            
            curr_duration = endTimestamp - startTimestamp - datetime.timedelta(seconds=1/(4*curr_sample_rate))
            curr_duration_ = end_index/(curr_sample_rate)
            
            assert abs(curr_duration.total_seconds()-curr_duration_)<1/curr_sample_rate
            
            #curr_start_time = curr_end_time - curr_duration
            sync_start_time = startTimestamp
        else:
            curr_duration = datetime.timedelta(seconds = (end_index)/curr_sample_rate)
            # in the case of q.station files, we need to rely on the file time for certain times
            sync_start_time = get_synchronized_time(curr_start_time, curr_file_time, curr_duration)
            
            sync_end_time = sync_start_time + curr_duration
            
        sync_start_time = np.datetime64(TC.to_storage_coord(sync_start_time))
        sync_end_time = np.datetime64(TC.to_storage_coord(sync_end_time))
        
        # if current file starts earlier than time_range[0] truncate some samples at the start
        if sync_start_time <= time_range[0]:
            truncate_time = np.timedelta64(time_range[0] - sync_start_time, 's')
            start_index = np.floor(truncate_time.astype('int') * curr_sample_rate).astype('int')
            new_start = sync_start_time + np.timedelta64(int(start_index/curr_sample_rate), 's')
            logger.debug('Truncating {} samples / {} seconds at the start of the file'.format(start_index, truncate_time.astype(int)))

        # if this is the first file and it starts later than time_range[0] then break because there is a gap before
        elif file_num == 0:
            if sync_start_time>time_range[0]:
                logger.debug('Gap before file. Skipping')
                return None
        # only one file that starts later than time_range 
        elif len(file_info.time) == 1:
            logger.warning('Bug: only one file that starts later than time_range ')
            new_start = sync_start_time.item()
        
        # if current file ends later than time_range[1] truncate some samples at the end
        if sync_end_time >= time_range[1]:
            truncate_time = np.timedelta64(sync_end_time - time_range[1] ,'s')
            end_index_ = np.ceil(((np.timedelta64(curr_duration,'s') - truncate_time).astype(int) * curr_sample_rate)).astype(int)
            end_index, end_index_ = end_index_, end_index-end_index_
            new_end = sync_end_time - np.timedelta64(int(end_index_/curr_sample_rate), 's')
            logger.debug('Truncating {} samples / {} seconds at the end of the file'.format(end_index_, truncate_time.astype(int)))
        
        # only one file that ends earlier than time_range
        elif len(file_info.time) ==1:
            logger.warning('Bug: only one file that ends earlier than time_range.')
            new_end =sync_end_time.item()
        
        # if last file ends earlier than time_range
        elif file_num == len(file_info.time)-1:
            logger.warning('Bug: last file ends earlier than time_range')
            new_end =sync_end_time.item()
            
        
        channel_index=[]
        for channel in channels:
            index = curr_headers.index(channel)
            channel_index.append(index)
            
        this_m = curr_measurement[start_index:end_index,channel_index]
        if this_m.shape[0]>0:
            all_measurements.append(this_m)
        else:
            #account for empty files
            all_measurements.append(np.zeros((0,len(channels))))
        
        # fill missing rows, except after the last one
        if gap_length > 0 and file_num != len(file_info.time)-1:
            rows = int(np.floor(gap_length))
            if rows > 0:
                curr_measurement = np.empty(( rows,len(channels)))
                curr_measurement[:]=np.NaN
                all_measurements.append(curr_measurement)
    
    logger.debug(all_measurements)
    #interleave consecutive files at_nan
    for meas_num in range(max(0,len(all_measurements)-1)):
        # check if nans in last row of current measurement 
        # and nans in complementary position in first row of next measurement
        
        # may be a length-0 file slice -> skip
        if not all_measurements[meas_num].shape[0]:
            continue
        if not all_measurements[meas_num+1].shape[0]:
            continue
        
        a = all_measurements[meas_num  ][-1,:]
        b = all_measurements[meas_num+1][ 0,:]
        
        if np.logical_or(np.isnan(a),
                       np.isnan(b)).any():
            
            b[np.isnan(b)] = a[np.isnan(b)]
            all_measurements[meas_num] = all_measurements[meas_num][:-1,:]
            
    this_measurement = np.vstack(all_measurements)
    
    this_sample_rate = this_measurement.shape[0]/np.timedelta64(new_end-new_start,'s').astype(int)
    # slice is empty
    if not this_measurement.shape[0]:
        return None
    
    # especially temperature measurements are taken only once in ten minutes (at some times)
    # we need to have some information about the time difference of the first and last sample from the start of the file
    # thus we upsample to an appropriate sample rate, s.t. these measurements can be used for temperature compensation
    if upsample_fs is not None:
        logger.info('Upsampling record from {} Hz to {} Hz'.format(this_sample_rate, upsample_fs))
        num_samples_new = round(duration.total_seconds()*upsample_fs)
        logger.debug('num_samples_new: {}'.format( num_samples_new))
        t_org = np.linspace(pd.Timestamp(new_start).value, pd.Timestamp(new_end).value, this_measurement.shape[0])            
        t_new  = np.linspace(pd.Timestamp(time_range[0]).value, pd.Timestamp(time_range[1]).value, num_samples_new)
        if this_measurement.shape[0]==1:
            new_m = np.zeros((num_samples_new,this_measurement.shape[1]))
            new_m[:,:]=this_measurement
        else:
            if this_measurement.shape[0]<=2:
                kind='linear'
            else:
                kind='quadratic'
            new_m=scipy.interpolate.interp1d(t_org, this_measurement, axis=0, bounds_error=False, fill_value='extrapolate', kind=kind)(t_new)
            
        this_measurement = new_m
        new_start = time_range[0]
        new_end = time_range[1]
        #sample_rate=upsample_fs
        this_sample_rate = this_measurement.shape[0]/(duration.total_seconds())

    logger.debug('Final length of slice: {} samples / {} seconds / {} channels'.format(this_measurement.shape[0], this_measurement.shape[0]/this_sample_rate, this_measurement.shape[1]))
    
    return new_start, list(channels), units, new_end, this_sample_rate, this_measurement

def get_slice_corrected(start_time: pd.Timestamp, duration: pd.Timedelta, 
                        quantity: str, file_info: xr.Dataset=None, **kwargs):
    st = start_time
    
    slice_name = '{}-{:02d}-{:02d}_{:02d}-{:02d}_{}.npz'.format(
        st.year, st.month, st.day, st.hour, st.minute, quantity)
    
    minutes = int(duration.total_seconds()/60)
    
    slice_root = os.path.join(_active_site.slice_root_path, f'{minutes}-minutes',
                              'slices_{}'.format(quantity),
                              '{}'.format(st.year), '{:02d}'.format(st.month))
    
    slice_path = os.path.join(slice_root, slice_name)
    
    if os.path.exists(slice_path):
        logger.info('Loading corrected signal slice for {} at {}'.format(quantity, start_time))
        try:
            
            in_dict = np.load(slice_path, allow_pickle=True)
            lstart_time = in_dict['start_time'].item()
            lheaders = list(in_dict['headers'])
            lunits = list(in_dict['units'])
            lend_time = in_dict['end_time'].item()
            lsample_rate = float(in_dict['sample_rate'])
            if not np.any([math.isclose(lsample_rate, fs, abs_tol=0.005) for fs in [1/60, 10, 20, 50, 100]]):
                logger.error('Sample Rate wrong! fs={}'.format(lsample_rate))
            lmeasurement = in_dict['measurement']
            
            ustrain=np.array(lunits)=='µstrain'
            lmeasurement[:,ustrain] *= 1e-6
            
            delta_t = (lend_time-lstart_time).total_seconds()
            
            if not math.isclose(delta_t, duration.total_seconds(), abs_tol = 0.1):
                logger.warning('Duration {} does not match given {}'.format((lend_time - lstart_time), duration))
            
            return lstart_time, lheaders, lunits, lend_time, lsample_rate, lmeasurement 
        except Exception as e:
            logger.exception(e)
            os.remove(slice_path)
        

    if not os.path.exists(slice_path) and file_info is None:
        logger.warning(slice_path + ' does not exist, file_info needed for creation of new slice!')    
        return None
 
    data_slice = get_slice(start_time, duration, quantity, file_info)

    logger.debug(f'Correcting slice for {quantity}: {start_time}')

    if data_slice is None:
        return None

    site = _active_site
    if site is not None:
        transform = site.transforms.get(quantity)
        if transform is not None:
            data_slice = transform(
                *data_slice,
                quantity=quantity,
                start_time_local=start_time,
                duration=duration,
                **kwargs,
            )
            if data_slice is None:
                return None

    start_time, headers, units, end_time, sample_rate, measurement = data_slice
    
    out_dict = {'start_time':start_time,
        'headers':headers,
        'units':units,
        'end_time':end_time,
        'sample_rate':sample_rate,
        'measurement':measurement
        }

    if not os.path.exists(slice_root):
        os.makedirs(slice_root)
    
    np.savez_compressed(slice_path, **out_dict)
    
    ustrain=np.array(units)=='µstrain'
    measurement[:,ustrain] *= 1e-6
    
    return start_time, headers, units, end_time, sample_rate, measurement 

def get_slice_preprocessed(start_time: pd.Timestamp, duration, quantity, file_info=None, **kwargs):
    '''
    remove channels, detrend, filter bandpass and decimate
    '''
    highpass = 0.1
    lowpass = 5
    target_fs = 10
    
    data_slice = get_slice_corrected(start_time, duration, quantity, file_info, **kwargs)

    if data_slice is None:
        return None

    actual_start_time, headers, units, end_time, sample_rate, measurement = data_slice
    
    if np.isnan(measurement).any():
        logger.warning("Measurement slice {}: {} contains nan. Skipping!".format(start_time, quantity))
        return None
    
    nyq = sample_rate/2
    dec_fact = round(sample_rate/target_fs)


    if 'strain' in quantity:
        site = _active_site
        keep = site.preproc_channels.get(quantity, []) if site is not None else []
        channel_selector = np.array([h in keep for h in headers])
        headers = [h for h in headers if h in keep]
        measurement = measurement[:, channel_selector]

    b, a = scipy.signal.butter(4, [highpass/nyq, lowpass/nyq], btype='band')
    
    ftype=scipy.signal.ltisys.dlti(b,a)
    meas_dec = scipy.signal.decimate(measurement, dec_fact, axis=0, ftype=ftype, zero_phase=True)
    
    measurement = meas_dec
    sample_rate = sample_rate/dec_fact      
    
    return actual_start_time, headers, units, end_time, sample_rate, measurement   


def get_modal_results(quantity: str, duration: pd.Timedelta, 
                      stats: xr.Dataset=None, create_new: bool=False, **kwargs):
    
    site = _active_site
    minutes = int(duration.total_seconds()/60)
    ds_path = os.path.join(site.db_root_path, f'{minutes}-minutes/', 'modal_{}.nc'.format(quantity))

    if not os.path.exists(ds_path):
        logger.warning(f'Path for modal does not exist {ds_path}.')

    if stats is None and create_new:
        raise RuntimeError('Statistics xarray has to be provided to run modal analysis.')

    if not create_new:
        stat_result = os.stat(ds_path, follow_symlinks=True)
        cache_key = f'{quantity}_modal'
        cached = _ds_cache.get(cache_key)
        if cached is not None and cached['mtime'] == stat_result.st_mtime:
            logger.info('Getting modal results for {} (cached)'.format(quantity))
            ds = cached['ds']
        else:
            logger.info('Getting modal results for {}'.format(quantity))
            ds = xr.open_dataset(ds_path, engine='h5netcdf')
            ds.load()
            ds.close()
            _ds_cache[cache_key] = {'ds': ds, 'mtime': stat_result.st_mtime}
    else:
        ds = create_modal_results(quantity, duration, stats, **kwargs)

    return ds

def create_modal_results(quantity: str, duration: pd.Timedelta, 
                         stats: xr.Dataset, chunksize: int=2, 
                         skip_existing: bool=True, check_errors: bool=True, 
                         filter_errors: bool=True, **kwargs):    
    # for the analysis accel and strain_rosettes should be error free
    # error has to be extended by checking range and kurtosis (similar to plot_stats() 
    # any error in the considered channels marks the respective slice as errorneous
    
    # stats/slices should already be aligned to the desired durations
    # all channels should be present in accels and strains -> No check on channels 
        
    # preprocessing: get_slice_preprocessed() 
    # describe: describe_stats() 
    # analyze with < OMA_Method > and postprocess with AOMA (E. Neu, Aachen): modal_analysis_single()

    logger.info('Creating modal results for {}'.format(quantity))

    # prepare modal analysis, config files, geometry, result_folder, etc.
    site = _active_site
    PreProcessSignals.load_measurement_file = dummy_load
    _conf_dir = os.path.join(site.modal_conf_dir, quantity)

    if check_errors:
        stats = check_and_mark_errors(stats)

    minutes = int(duration.total_seconds()/60)

    process_ds_path = os.path.join(site.db_root_path, f'{minutes}-minutes/', f'modal_{quantity}.{_pid}.nc')
    master_ds_path = os.path.join(site.db_root_path, f'{minutes}-minutes/', 'modal_{}.nc'.format(quantity))
    
    if os.path.exists(process_ds_path):
        process_ds =  xr.open_dataset(process_ds_path, engine='h5netcdf')
        process_ds.load()
        process_ds.close()
    else:
        process_ds = xr.Dataset() 
        process_ds.to_netcdf(process_ds_path, engine='h5netcdf')
        process_ds.close()
    
    if os.path.exists(master_ds_path):
        master_ds = xr.open_dataset(master_ds_path, engine='h5netcdf')
        master_ds.load()
        master_ds.close()
    else:
        master_ds = xr.Dataset()
        
        
        
    # cases:
    # distributed processing -> master_ds inconsistent, changing when worker updates it
    # single processing -> master_ds consistent
    
    # case                      time_iterator                                master_ds
    # proc. of all             -> use stats.time                             -> inconsistent 
    # proc. [dtstart .. (until)] -> stats.time (filtered)                      -> inconsistent
    # proc. missing            -> difference master_ds.time and stats_time - -> needs consistent
    #
    distributed_processing=True
    
    dtstart=kwargs.pop('dtstart', None)
    until = kwargs.pop('until', None)
    missing = kwargs.pop('missing', False)
    if dtstart is not None:
        time_iterator = stats.time[stats.time>=dtstart]
        if until is not None:
            time_iterator = time_iterator[time_iterator<=until]
    elif missing and 'time' in master_ds.dims:
        distributed_processing =False
        # drop nan from both ds
        # compute the set difference
        
        # do not dropna, because, then it will try all errorneous files again every time
        #master_ds = master_ds.dropna(dim='time', how='all')
        stats_ds = stats.dropna(dim='time', how='all')
        time_iterator = np.setdiff1d(stats_ds.time, master_ds.time)
        errorneous_timestamps = stats_ds.time[np.logical_or(
            stats_ds['error'].any(dim='channels'), stats_ds.length.isnull())]
        time_iterator = np.setdiff1d(time_iterator, errorneous_timestamps)
    else:
        time_iterator = stats.time.values
    
    
    num_workers = kwargs.get('num_workers',1)
    this_worker = kwargs.get('this_worker',1)
    jobsize=int(np.ceil(len(time_iterator)/num_workers))
    start = int((this_worker-1)%num_workers)
        
    if not distributed_processing and num_workers>1:
        raise RuntimeError("You can't process 'only missing' data using multiple workers")

        
         
    logger.debug(jobsize)
    i=1
    for start_time in time_iterator[start*jobsize:(start+1)*jobsize]:

        logger.debug('')
        start_time_naive = pd.Timestamp(start_time)
        start_time = TC.to_local(start_time)
        if pd.isnull(start_time):
            continue
        # if not (i+1)%50: print('.',end='', flush=True)
        # if not (i+1)%2500: print('\n')

        this_stats = stats.sel(time=start_time_naive)
        if this_stats['error'].any():
            logger.warning('Error in slice {}: {}'.format(start_time, this_stats['error'].data))
            continue
        
        if 'time' in master_ds.coords:
            if (start_time.to_datetime64()==master_ds.coords['time']).any():
                if skip_existing: 
                    # other checks may be added here
                    continue
                else: 
                    logger.info(f'{str(start_time)} already present in master dataset. Updating!')
                    # other checks may be added here
            
        if 'time' in process_ds.coords:
            if (start_time.to_datetime64()==process_ds.coords['time']).any():
                if skip_existing: 
                    # other checks may be added here
                    continue
                else:
                    logger.info(f'{str(start_time)} already present in process dataset. Updating!')
                    process_ds = process_ds.where((start_time.to_datetime64()!=process_ds.coords['time']), drop=True)
                    # other checks may be added here
        
        this_ds = xr.Dataset()
        
        if this_stats.length.isnull():
            continue
        
        # duration=pd.Timedelta(seconds=(this_stats.length/this_stats.sample_rate).variable.data.item())
        
        try:
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                data_slice = get_slice_preprocessed(start_time, duration, quantity, **kwargs)
        except Exception as e:
            logger.warning('Exception while trying to get preprocessed slice: ')
            logger.warning(e)
            data_slice=None

        if data_slice is None:
            this_ds.coords['time'] = [start_time.to_datetime64()]
        else:
            _actual_start_time, headers, _units, _end_time, sample_rate, measurement = data_slice
            
    
            this_ds['num_channels'] = (['time'], [measurement.shape[1]])#:len(headers), 
            this_ds['sample_rate'] = (['time'], [sample_rate])#:sample_rate,
            this_ds['length']= (['time'], [measurement.shape[0]])
            
            this_dict = describe_stats(measurement, headers, quantity)
            
            
            for var_name in ['mean','min','max','var','skewness','kurtosis','q05','q50','q95','rms','error']: 
                try:# may not be present for every file
                    this_ds[var_name] = (['time','channels'], [this_dict[var_name]])
                except KeyError as e:
                    logger.warning(e)
                    continue
                except Exception:
                    logger.warning('error')
                    
            this_ds.coords['channels'] = np.array(headers,dtype=str)
            this_ds.coords['time'] = [start_time.to_datetime64()]
            
            try:
                #remove errorneous channels
                #if this_ds['error'].any():
                #    pass
                
                now=time.time()
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    results = modal_analysis_single(start_time, data_slice, quantity,
                                    duration)
                n, f, std_f, d, std_d, MPC, _MP, MPD, _MC, msh, s_vals_psd = results
                logger.info('Modal Analysis took {:.2f} s to return {} modes for : {}'.format(time.time()-now, len(n), start_time))
                
                this_dict = describe_stats(s_vals_psd.T)
                for var_name in ['mean','min','max','var','skewness','kurtosis','q05','q50','q95','rms']: 
                    try:# may not be present for every file
                        this_ds[var_name+'_svd_psd'] = (['time','channels'], [this_dict[var_name]])
                    except Exception as e:
                        logger.exception(e)
                energy_f = np.sum(np.power(np.abs(s_vals_psd),2),axis=-1)/s_vals_psd.shape[-1]
                this_ds['energy_svd_psd'] = (['time','channels'],[energy_f])
    
                this_ds['num_modes'] = (['time'], [len(n)])
                this_ds['model_orders'] = (['time','modes'], [n])  
                this_ds['frequencies'] = (['time','modes'], [f])  
                this_ds['std_frequencies'] = (['time','modes'], [std_f])              
                this_ds['damping'] = (['time','modes'], [d])   
                this_ds['std_damping'] = (['time','modes'], [std_d]) 
                this_ds['MPC'] = (['time','modes'], [MPC])  
                this_ds['MPD'] = (['time','modes'], [MPD])
                #this_ds['modal_contributions'] = (['time','modes'], [MC])  
                this_ds['modeshapes'] = (['time','modes','channels'], [msh])            
                
                # assignment to named modes will not be done here, as this
                # would require the rejection of a large amount of modes
                # and this in turn would bias any further analyzes
                # see function.... for further explanations
                this_ds.coords['modes'] = list(range(len(n)))
                
            except Exception as e:
                logger.exception('Error. Continuing')
                logger.exception(e)
            
        process_ds = process_ds.combine_first(this_ds)
        # logger.info('Succes: {}, Length: {}'.format((start_time.to_datetime64()==process_ds.coords['time']).any().item(), len(process_ds.time)))
        i+=1
        
        if i>0 and not i%chunksize:
            # process_ds should not have changed on disk during processing
            process_ds = save_ds(process_ds, process_ds, process_ds_path, what='modal')

    # process_ds should not have changed on disk during processing
    process_ds = save_ds(process_ds, process_ds, process_ds_path, what='modal')

    # master_ds almost certainly has changed on disk if multiple workers were processing files
    master_ds = save_ds(process_ds, master_ds, master_ds_path, reload_current = True, what='modal')

    logger.debug(f'Removing temporary dataset at {process_ds_path}')
    os.remove(process_ds_path)

    return master_ds

def modal_analysis_single(start_time, data_slice, quantity, duration):
    
    site = _active_site
    conf_dir = os.path.join(site.modal_conf_dir, quantity)
    st = start_time

    skip_existing = True
    save_results = True
    interactive = False

    minutes = int(duration.total_seconds()/60)

    result_folder = os.path.join(site.slice_root_path,
                                 f'{minutes}-minutes',
                                 'modal_{}'.format(quantity),
                                 '{}'.format(st.year),
                                 '{:02d}'.format(st.month))
        
    nodes_file = os.path.join(conf_dir, 'nodes')
    lines_file =os.path.join(conf_dir, 'lines')
    master_slaves_file=os.path.join(conf_dir, 'master_slaves')

    geometry_data = GeometryProcessor.load_geometry(nodes_file, lines_file, master_slaves_file)    
    
    _conf_file = os.path.join(conf_dir, 'setup_info')
    _chan_dofs_file = os.path.join(conf_dir, 'channel_dofs')
    ssi_file = os.path.join(conf_dir, 'ssi_config')
    _plscf_file = os.path.join(conf_dir, 'plscf_config')
    
    fname = os.path.join(result_folder,
        '{:02d}_{:02d}-{:02d}_{:02d}-{:02d}_prep_data.npz'.format(
            st.year, st.month, st.day, st.hour, st.minute))
    
    if not os.path.exists(fname) or not skip_existing:
        start_time, headers, _units, _end_time, sample_rate, measurement \
            = data_slice

        site = _active_site
        if site is None or quantity not in site.setup_prep:
            raise RuntimeError(
                f"No setup_prep registered for quantity '{quantity}'. "
                "Import the appropriate site module before running modal analysis."
            )
        ref_channels, accel_channels, disp_channels, chan_dofs_dict = \
            site.setup_prep[quantity](headers)
            
        prep_data = PreProcessSignals(measurement, sample_rate, 
                        ref_channels=ref_channels, accel_channels=accel_channels,
                        disp_channels= disp_channels, channel_headers=headers,
                        start_time=start_time)
        
        chan_dofs = [[chan]+chan_dofs_dict[header] for chan, header in enumerate(headers)]
        prep_data.add_chan_dofs(chan_dofs)
        s_vals_psd = prep_data.sv_psd(1444, method='blackman-tukey', refs_only=False)
        _freqs = prep_data.freqs
        if save_results:
            prep_data.save_state(fname)
    else:
        prep_data = PreProcessSignals.load_state(fname)
        s_vals_psd = prep_data.sv_psd(1444, method='blackman-tukey', refs_only=False)
        _freqs = prep_data.freqs

    
    fname = os.path.join(result_folder,
        '{:02d}_{:02d}-{:02d}_{:02d}-{:02d}_modal_data.npz'.format(
            st.year, st.month, st.day, st.hour, st.minute))
    
    if not os.path.exists(fname) or not skip_existing:
         
        modal_data = VarSSIRef.init_from_config(ssi_file, prep_data)   
        #modal_data= SSIDataMC.init_from_config(ssi_file, prep_data)
        #modal_data = PLSCF.init_from_config(plscf_file, prep_data)

        if save_results: 
            modal_data.save_state(fname)
    else:
        #modal_data=SSIDataMC.load_state(fname, prep_data)
        modal_data=VarSSIRef.load_state(fname, prep_data)
      
    fname = os.path.join(result_folder,
        '{:02d}_{:02d}-{:02d}_{:02d}-{:02d}_stabil_data.npz'.format(
             st.year, st.month, st.day, st.hour, st.minute))
    
    if not os.path.exists(fname) or not skip_existing:
        stabil_calc = StabilCluster(modal_data)    
        
    else:
        try:
            stabil_calc= StabilCluster.load_state(fname, modal_data)
        except Exception as e:
            logger.warning(e)
            stabil_calc = StabilCluster(modal_data)   
            
    if stabil_calc.state==5:
        save_results=False
    
    if stabil_calc.state<1:
        stabil_calc.calculate_soft_critera_matrices()
    if stabil_calc.state<2:            
        stabil_calc.calculate_stabilization_masks()
    if stabil_calc.state<3:
        stabil_calc.automatic_clearing()
    if stabil_calc.state<4:
        stabil_calc.automatic_classification()
    if stabil_calc.state<5:
        stabil_calc.automatic_selection()        

    figure_name = os.path.join(result_folder, 
        '{:02d}_{:02d}-{:02d}_{:02d}-{:02d}_stabil.png'.format(
             st.year, st.month, st.day, st.hour, st.minute))
    
    if not os.path.exists(figure_name) or not skip_existing or interactive:
        stabil_plot=StabilPlot(stabil_calc)
        stabil_plot.plot_stabil(name='plot_pre')
        stabil_plot.plot_stabil(name='plot_autoclear')
        stabil_plot.plot_stabil(name='plot_autosel')
        stabil_plot.plot_sv_psd(True, 1444)
        stabil_plot.update_xlim((0.2,4))
        stabil_plot.save_figure(figure_name)
      
    if interactive and start_stabil_gui is not None:
        start_stabil_gui(stabil_plot, modal_data, geometry_data,
                         prep_data=prep_data)
        
    if save_results: 
        stabil_calc.save_state(fname)       
    results = list(stabil_calc.return_results())+[s_vals_psd]
    return results
        
def dummy_load(meas_file, **kwargs):
    headers = kwargs['headers']
    units = kwargs['units']
    start_time = kwargs['start_time']
    sample_rate = kwargs['sample_rate']
    measurement = kwargs['measurement']
    
    return headers, units, start_time, sample_rate, measurement


def split_modepairs(modal):
    '''
    takes all frequencies, that occur in pairs in a given frequency range
    reorders and assigns all other variables correspondingly
    '''
    site = _active_site
    modal_bands = site.modal_bands if site is not None else []

    modal_sorted = xr.Dataset()

    for modepair, f_range in enumerate(modal_bands):
        new_modal = xr.Dataset()
        
        # filter frequency range
        frequencies = modal['frequencies']
        f_ind = np.logical_and(frequencies>=f_range[0],frequencies<=f_range[1])
        # filter "exactly two modes"
        f_ind = f_ind.where(f_ind.sum(dim='modes')==2, False)
        frequencies = frequencies.where(f_ind)

        for name,data in modal.variables.items():
            
            if name in data.dims:continue
            
            if 'modes' in data.dims:
                #pass   
                
                mode_ind_0 = frequencies==frequencies.min(dim='modes')
                data_new_0 = data.where(mode_ind_0).min(dim='modes').expand_dims(dim='modes').rename(name)
                data_new_0.coords['modes']=('modes',[modepair*2 + 0])
                
                mode_ind_1 = frequencies==frequencies.max(dim='modes')
                data_new_1 = data.where(mode_ind_1).min(dim='modes').expand_dims(dim='modes').rename(name)
                data_new_1.coords['modes']=('modes',[modepair*2 + 1])
                
                data_new = xr.merge([data_new_0,data_new_1])
            else:
                data_new = xr.Dataset({name:data})

            new_modal = xr.merge([new_modal, data_new])
        
        new_modal.coords['channels'] = modal.coords['channels']
        modal_sorted = xr.merge([modal_sorted, new_modal])
    logger.debug(modal_sorted)
    return modal_sorted

def merge_xarrays(path, quantity, what='modal',delete=None):
    
    fpath = os.path.join(path,'result_db/{}_{}.nc'.format(what,quantity))
    filelist = glob.glob(os.path.join(path,'result_db/{}_{}.*.nc'.format(what,quantity)))
    if not filelist:
        logger.warning('List of xarrays is empty. Returning.')
        return
    if os.path.exists(fpath):
        
        if delete is None:
            delete = input('Path {} exists. {} files can be merged. Do you want to delete (y/n) or also merge (u) file? '.format(fpath, len(filelist)))
        assert delete in ['y', 'u', 'n']
            
        if delete =='y':
            os.remove(fpath)
        elif delete == 'u':
            new_path = fpath.strip('.nc')+'_old.nc'
            logger.debug('cp {} {}'.format(fpath, new_path))
            if os.system('cp {} {}'.format(fpath, new_path)) == 0:
                filelist += [new_path]
                os.remove(fpath)
            else:
                return
        elif delete=='n':
            i=0
            while True:
                new_path = fpath.strip('.nc')+'_old_{}.nc'.format(i)
                if os.path.exists(new_path):
                    i+=1
                else:
                    os.rename(fpath,new_path)
                    break
            del i
        else:
            return
    logger.debug(filelist)
    arraylist = []
    for file in filelist:
        try:
            if what=='modal':
                ds =  xr.open_dataset(file, engine='h5netcdf')
            elif what=='stats' or what=='file_info':
                ds =  xr.open_dataset(file)
            
            ds.load()
            ds.close()
            logger.debug(ds.dims.get('time',0))
            arraylist.append(ds)
        except:
            logger.exception(file)
            raise
    
    new_ds = xr.Dataset()
    for i,ds in enumerate(arraylist):
        now = time.time()
        
        try:
            new_ds = new_ds.combine_first(ds)
        except:
            logger.exception(filelist[i])
            raise
        logger.debug('{} {} s'.format(i, time.time()-now))
        ds.close()
    logger.info('Merged {} files.'.format(len(filelist)))
    load_ds = None
    if what=='modal':
        new_ds.to_netcdf(fpath, engine='h5netcdf')
        load_ds = xr.open_dataset(fpath, engine='h5netcdf').load()
    elif what=='stats' or what=='file_info':
        new_ds.to_netcdf(fpath, format='NETCDF4')
        load_ds = xr.open_dataset(fpath).load()
    logger.debug(load_ds)
    logger.info('Saved dataset to {}'.format(fpath))
    #delete = input('Merged ds has {} entries, re-loaded dataset has {} entries. Do you want to delete all source files? (y/n)'.format(new_ds.dims.get('time',0),load_ds.dims.get('time',0)))   
    if delete in ['y', 'u']:
        for file in filelist:
            os.rename(file, file+'.old')
            #os.remove(file)
            logger.info('Deleted {}'.format(file))
def main():
    
    logger.setLevel(logging.INFO)
    # sys.argv = [..., this_worker, num_workers, duration_selector, q_selector]
    #sys.argv+=[10,20,1,3]
    if len(sys.argv)>2: 
        num_workers = int(sys.argv[2])
        if num_workers>1:
            time.sleep(np.random.random()*num_workers)
        this_worker = int(sys.argv[1])
    else:
        num_workers=1
        this_worker=1
    
    if len(sys.argv)>3: duration_selector = int(sys.argv[3])
    else: duration_selector = 3
        
    duration = pd.Timedelta(minutes=[10,30,60,120][duration_selector])
    minutes = int(duration.total_seconds()/60)
    site = _active_site
    _db_path = os.path.join(site.db_root_path, f'{minutes}-minutes/')

    if len(sys.argv) > 4: q_selector = int(sys.argv[4])
    else: q_selector = 1

    quantities = [
        'accel',          #0
        'wind',           #1
        'temp',           #2
        'strain_rosettes',#3
        #'strain_bolts'
        ][q_selector:q_selector+1]

    for quantity in quantities:

        origin = site.origins[quantity]
        _subpath = site.subpaths[origin]
        logger.info('Quantity: {}, Duration: {}'.format(quantity, minutes))

        if 0:
            file_contents = read_file(os.path.join(site.file_root_path, site.subpaths[origin], 'Wind_kontinuierlich__1_2018-06-13_15-00-00_000000.csv.bz2'))
            logger.debug(file_contents)
            meas = file_contents[-1]
            logger.debug(meas.shape)
            ts = meas[:, 0]
            logger.debug(ts[-1]-ts[0])
            for i in range(meas.shape[0]):
                print(ts[i+1]-ts[i])
                if (ts[i+1]-ts[i]) != 1:
                    print(i)
                    break
            return

        if 0: # create
            file_info = get_file_info(origin, create_new=True, skip_existing=True, reduced=False)
            return
        else: # get
            file_info = get_file_info(origin, create_new=False)

        if 0:
            data_slice = get_slice_corrected(pd.Timestamp('2018-02-27 20:00', tz='Europe/Berlin'),
                                        duration,
                                        quantity,
                                        file_info,
                                        file_info_temp=get_file_info(site.origins['temp']))
            print(data_slice[:-1])
            print(data_slice[-1].shape)
            print(np.mean(data_slice[-1], axis=0))
            _actual_start_time, headers, _units, _end_time, _sample_rate, measurement = data_slice

            this_dict = describe_stats(measurement, headers, quantity)
            print(this_dict)
            return

        if 1:
            if 'strain' in quantity:
                file_info_temp = get_file_info(site.origins['temp'])
                
                stats = get_stats(quantity, duration,file_info, 
                                  create_new=True, skip_existing=True, chunksize=500, 
                                  file_info_temp=file_info_temp,
                                  num_workers=num_workers, this_worker=this_worker)
            else:
                stats = get_stats(quantity, duration, file_info, 
                                  create_new=True, skip_existing=True, chunksize=500, 
                                  num_workers=num_workers, this_worker=this_worker)
                
            return
        else:
            stats = get_stats(quantity, duration, )
        
        
        if 0 and quantity in ['accel', 'strain_rosettes']:
            _modal = get_modal_results(quantity, duration, stats,
                                      skip_existing=True, create_new=True, filter_errors=False,
                                      chunksize=20, num_workers=num_workers, this_worker=this_worker)
            return
        

    return
    
                    
if __name__ == '__main__':
    main()
