'''
A data-organization, -storage and analysis scheme for monitoring system “Geyer”

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
import warnings
import contextlib
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import re
import pytz
import datetime
import dateutil
import time
import tzlocal

berlin_dst=pytz.timezone('Europe/Berlin') # cet/cest

from time_convention import TC

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
from pyOMA.core.StabilDiagram import StabilCalc, StabilCluster, StabilPlot
try:
    from pyOMA.GUI.StabilGUI import start_stabil_gui
except ImportError:
    start_stabil_gui = None
from pyOMA.core.PlotMSH import ModeShapePlot
# from GUI.PlotMSHGUI import start_msh_gui
from pyOMA.core.VarSSIRef import VarSSIRef
#from PLSCF import PLSCF
#from SSIData import SSIDataMC

import config
reader_tz = tzlocal.get_localzone()
        
        
def get_file_list(origin, reduced=False, file_info = None):
    '''
    read in all files from path
    origin may be 'q_station' or 'labview'
    return list of files, creation dates, file sizes
    '''
    path = os.path.join(config.file_root_path, config.subpaths[origin])
    path = os.path.normpath(path)
    if not os.path.exists(path):
        raise RuntimeError(f"{path} does not exist.")
    
    if origin =='accel':
        file_list = glob.glob(os.path.join(path,'Accel_continuously__*'))
        if not reduced:
            file_list += glob.glob(os.path.join(path,'Alle_3h_00_00__1_2015-04-*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_00_00__1_2015-05-0*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_00_00__3_2015-04-*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_00_00__3_2015-05-0*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_03_00__2_2015-04-*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_03_00__2_2015-05-0*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_03_00__4_2015-04-*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_03_00__4_2015-05-0*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_06_00__3_2015-04-*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_06_00__3_2015-05-0*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_06_00__5_2015-04-*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_06_00__5_2015-05-0*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_09_00__4_2015-04-*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_09_00__4_2015-05-0*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_09_00__6_2015-04-*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_09_00__6_2015-05-0*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_12_00__5_2015-04-*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_12_00__5_2015-05-0*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_12_00__7_2015-04-*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_12_00__7_2015-05-0*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_15_00__6_2015-04-*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_15_00__6_2015-05-0*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_15_00__8_2015-04-*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_15_00__8_2015-05-0*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_18_00__7_2015-04-*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_18_00__7_2015-05-0*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_18_00__9_2015-04-*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_18_00__9_2015-05-0*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_21_00__8_2015-04-*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_21_00__8_2015-05-0*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_21_00__10_2015-04-*'))
            file_list += glob.glob(os.path.join(path,'Alle_3h_21_00__10_2015-05-0*'))
            file_list += glob.glob(os.path.join(path,'Wind_kontinuierlich__*'))
    elif origin =='wind':
        file_list = glob.glob(os.path.join(path,'Wind_continuously__*'))
        if not reduced:
            file_list += glob.glob(os.path.join(path,'Wind_kontinuierlich__*'))
    elif origin=='temp':
        file_list = glob.glob(os.path.join(path,'Temp_continuously__*'))
        if not reduced:
            file_list += glob.glob(os.path.join(path,'Temp_konti_*'))
    elif origin=='strain':
        file_list = []
        paths = [path]
        if not reduced: paths.append(os.path.join(path,'binary_files_unusable'))
        for path_2 in paths:
            for filename in os.listdir(path_2):
                file,ext = os.path.splitext(filename)
                if ext == '.npz': continue
                if ext == '.bz2': file, ext = os.path.splitext(file)
                
                if file.endswith('spec'): continue
                
                if ext =='.txt':
                    file = re.sub(r"strain-[1-4]","",file)
                    if not file.endswith('_'): continue
                    if os.path.join(path_2,file) in file_list:
                        continue
                    file_list.append(os.path.join(path_2,file))
                elif ext == '.bin':
                    file_list.append(os.path.join(path_2,filename))
                
    else:
        RuntimeWarning('origin was neither accel nor wind nor strain nor temp. filelist is empty')
        file_list = []
        
    if reduced and file_info is not None:
        
        filename_list = [os.path.basename(file) for file in file_list]
        dset = set(filename_list).difference(file_info['file_name'].variable.data)
        logger.debug('{} {}'.format(len(dset), dset))
        file_list = [os.path.join(path, filename) for filename in dset]
        
    return  file_list

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
    
    for f_dict in config.file_cache:
        if f_dict.get('path','')==path:
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
        
        config.file_cache.append({'path':path,
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
     
    config.file_cache.append({'path':path,
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
    # ds = create_file_info(origin, **kwargs)
    ds_path = os.path.join(config.db_root_path, f'file_info_{origin}.nc')
    if not os.path.exists(ds_path):
        ds = xr.Dataset() 
        ds.to_netcdf(ds_path, format='NETCDF4')
        ds.close()
        logger.warning(f"DB Path {ds_path} was recreated.")
        # raise RuntimeError(f'Path for file_info does not exist {ds_path}.')
        return None

    
    if not create_new:
        stat_result = os.stat(ds_path, follow_symlinks=True)
        if config.ds_cache is not None and config.ds_cache[f'{origin}_file_info']['mtime']==stat_result.st_mtime:
            logger.info('Getting file info for {} (cached)'.format(origin))
            ds = config.ds_cache[f'{origin}_file_info']['ds']
        else:
            logger.info('Getting file info for {}'.format(origin))
            ds =  xr.open_dataset(ds_path)
            ds.load()
            ds.close()
            if config.ds_cache is not None:
                 config.ds_cache[f'{origin}_file_info']['ds'] =  ds
                 config.ds_cache[f'{origin}_file_info']['mtime'] = stat_result.st_mtime
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
    '''
    checks the provided time stamps and returns synchronized starting time
    
    
    local time: time of the measurement device (Q.Station, Labview/Illumisense PC)
    remote time: time of the storage device (Q.Station, PC)
    
    differences in local and remote time:
     - clock not set
     - different time zones (UTC, MEZ, MESZ)
     - day light savings
     - clock drifts
     
     Over the operating period, the settings have changed multiple times, therefore 
     time differences are varying
     
    start time is local time with local time zone
    file_time is when the last bit was written, it is local or remote time, with daylight saving
    
    duration is num_samples/sample_rate
    ideally: start_time + duration = file_time
    
    due to the time difference this does not hold true
    
    for a visual reference run the function 'inspect_time_shifts()'
    
     '''
    
    if start_time < berlin_dst.localize(datetime.datetime(2016,12,15)):
        # before Strain recordings were started no synchronization is needed
        return start_time
    elif start_time < berlin_dst.localize(datetime.datetime(2018,1,25)): # actually 8:25 o'clock
        # before NTC was set up, only reliable time information is the file's timestamp
        # time stamp covers daylight saving changes
        return file_time - duration
    elif start_time < berlin_dst.localize(datetime.datetime(2019,4,30)):# or maybe 2019,4,25
        # before PC was set to UTC it was running two hours ahead of the controller
        # daylight savings were disabled before, last change happened on 2017-10-29 03:00 to 01:00
        # this is already covered in fbg_strain_reader.localize()
        return start_time
    else:
        return start_time


    
def create_file_info(origin: str, chunksize: int=50, skip_existing: bool=True, **kwargs):
    '''
    file_name, file_size, file_creation_time, num_channels, headers, units, sample_rate, type (qstation, labview), start_time,<- 1st run
    length (in samples), per channel: {errors, mean, min, max, var, skewness, kurtosis, q05, q95, q50, rms} <- 1st run
    next_file, gap_length (in_samples) <- 2nd run
    '''
    
    logger.info('Creating file info for {}'.format(origin))   
    ds_path = os.path.join(config.db_root_path, 'file_info_{}.nc'.format(origin))
    
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
            
            for var_name, value in this_dict.items():
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
            this_range = config.ranges.get(header, None)
            
            suffix = header.replace('Wr','')
             
            if 'Wx'+suffix in headers and 'Wy'+suffix in headers:
                Wx = measurement[:,headers.index('Wx'+suffix)]
                Wy = measurement[:,headers.index('Wy'+suffix)]
                
                mean = orthogonal_lsq(xy=(Wx,Wy))
                
                Wr = np.copy(measurement[:,channel])
                Wr -= mean                    
                Wr[Wr<-180] += 360
                Wr[Wr>180] -= 360        
                Wr += mean
                
                nobs, (min_,max_), _, variance, skewness, kurtosis = scipy.stats.describe(Wr,nan_policy='omit')
                q05,q50,q95 = np.nanpercentile(Wr, q=[5,50,95])
                rms = np.sqrt(np.nanmean(np.square(Wr)))

            else:
                nobs, (min_,max_), mean, variance, skewness, kurtosis = scipy.stats.describe(measurement[:,channel],nan_policy='omit')
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
            if quantity in ['accel','strain_rosettes']:
                if kurtosis>5:
                    this_dict['error'][channel]=True
                if kurtosis<-2:
                    this_dict['error'][channel]=True
                    
                    
                    
                    
            
        except Exception as e:
            raise
            logger.warning(e)
            this_dict['error'][channel]=True   
    
    return this_dict



def check_and_mark_errors(ds, new = True, check_kurtosis = False):
    '''
    checks and marks errors in the dataset ds
    new: discard existing error flags and only use newly created error flags
    
    returns: dataset ds with updated error dataarray
    '''
    
    for channel in ds.channels:
        this_range = config.ranges.get(channel.variable.item(),None)
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
    
    minutes = int(duration.total_seconds()/60)
    ds_path = os.path.join(config.db_root_path, f'{minutes}-minutes/', 'stats_{}.nc'.format(quantity))
    
    if not os.path.exists(ds_path):
        logger.warning(f'Path for stats does not exist {ds_path}.')
        
        
    if file_info is None and create_new:
        raise RuntimeError('File info xarray has to be provided to create a statistical description')
        
    if not create_new:
        stat_result = os.stat(ds_path, follow_symlinks=True)
        if config.ds_cache is not None and config.ds_cache[f'{quantity}_stats']['mtime']==stat_result.st_mtime:
            logger.info('Getting statistics for {} (cached)'.format(quantity))
            ds = config.ds_cache[f'{quantity}_stats']['ds']
        else:
            
            logger.info('Getting statistics for {}'.format(quantity))
            ds =  xr.open_dataset(ds_path)
            ds.load()
            ds.close()
            if config.ds_cache is not None:
                 config.ds_cache[f'{quantity}_stats']['ds'] =  ds
                 config.ds_cache[f'{quantity}_stats']['mtime'] = stat_result.st_mtime
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
    
    process_ds_path = os.path.join(config.db_root_path, f'{minutes}-minutes/', 'stats_{}.{}.nc'.format(quantity,config.pid))
    master_ds_path = os.path.join(config.db_root_path, f'{minutes}-minutes/', 'stats_{}.nc'.format(quantity))
    
    if os.path.exists(process_ds_path):
        process_ds =  xr.open_dataset(process_ds_path)
        process_ds.load()
        process_ds.close()
    else:
       process_ds = xr.Dataset() 
       os.makedirs(os.path.join(config.db_root_path, f'{minutes}-minutes/', ), exist_ok=True)
       process_ds.to_netcdf(process_ds_path, format='NETCDF4')
       process_ds.close()
       
    if os.path.exists(master_ds_path):
        master_ds = xr.open_dataset(master_ds_path)
        master_ds.load()
        master_ds.close()
    else:
        master_ds = xr.Dataset()
        
    origin= config.origins[quantity]
    dtstart = config.dtstarts[origin]
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
                            slice = get_slice_corrected(start_time, duration, quantity, file_info, **kwargs)
                            logger.setLevel(old_level)
                        except Exception as e:
                            logger.exception(e)
                            slice=None
                        if slice is not None:
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
            slice = get_slice_corrected(start_time, duration, quantity, file_info, **kwargs)
            logger.setLevel(old_level)
        except Exception as e:
            logger.exception(e)
            slice=None
            
        if slice is None:
            logger.debug('Returned Slice is empty. Skipping!')
            
        else:
            actual_start_time, headers, units, end_time, sample_rate, measurement = slice
            
            this_ds['num_channels'] = (['time'], [measurement.shape[1]])#:len(headers), 
            this_ds['sample_rate'] = (['time'], [sample_rate])#:sample_rate,
            this_ds['length']= (['time'], [measurement.shape[0]])
            this_dict = describe_stats(measurement, headers, quantity)

            for var_name, value in this_dict.items():
                this_ds[var_name] = (['time','channels'], [this_dict[var_name]])

            this_ds.coords['channels'] = np.array(headers,dtype=str)

        # this_ds.coords['time'] = [np.asarray(start_time, dtype='datetime64[ns]')]
        this_ds.coords['time'] = [np.asarray(start_time.to_datetime64())]

        process_ds = process_ds.combine_first(this_ds)
        logger.debug('Success: {}, Length: {}'.format((start_time.to_datetime64()==process_ds.coords['time']).any().item(), len(process_ds.time)))
        
        if i>0 and not i%chunksize:
            # process_ds should not have changed on disk during processing
            process_ds = save_ds(process_ds, process_ds, process_ds_path, what='stats')
    else:
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
            dupes, ind_new, ind_current = np.intersect1d(new_ds.time, current_ds.time, assume_unique=True, return_indices=True)
            len_before = len(current_ds.time)
            current_ds = current_ds.drop_sel(time=dupes)
            logger.debug(f'dropped {len_before - len(current_ds.time)} results that were already present in the db.')
            current_ds = current_ds.combine_first(new_ds)
            '''
            For datasets, ds0.combine_first(ds1) works similarly to xr.merge([ds0, ds1]), 
            except that xr.merge raises MergeError when there are conflicting values in 
            variables to be merged, whereas .combine_first defaults to the calling object's values.
            '''
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
    file_info['gap_length'] = TC.gap_lengths(
        file_info['start_time'], file_info['duration'], file_info['sample_rate']
    )

def get_slice(start_time, duration , quantity, file_info, upsample_fs=None):
    '''
    channels 'Tagessekunden' and 'Time' are always dropped
    '''
    
    time_range = (start_time.to_datetime64(), (start_time + duration).to_datetime64())
    
    channels_required = config.all_channels[quantity]
    origin = config.origins[quantity]
    subpath = config.subpaths[origin]
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
        channels_to_drop = list(channels_in_file.difference(channels_required + config.optional_channels.get(quantity,[])))
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
        sample_rate = file_info.isel(time=file_num)['sample_rate']
        
        logger.debug(file.item())
        
        filename = file.item()
        if 'Peaks' in filename: filename= os.path.join('binary_files_unusable',filename)
        
        file_path = os.path.join(config.file_root_path, config.subpaths[origin], filename)
        
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f): 
            file_contents = read_file(file_path)
            
        if file_contents is None:
            logger.warning('File unreadable: {}'.format(file_path))
            return None
        curr_file_time, curr_file_size, curr_headers, curr_units, curr_start_time, curr_sample_rate, curr_measurement = file_contents
        
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
    
    slice_root = os.path.join(config.slice_root_path, f'{minutes}-minutes',
                              'slices_{}'.format(quantity),
                              '{}'.format(st.year),'{:02d}'.format(st.month))
    
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
 
    slice = get_slice(start_time, duration, quantity, file_info)
    
    logger.debug(f'Correcting slice for {quantity}: {start_time}')
    
    if slice is None:
        return None
    
    if quantity == 'wind':
        slice = wind_transform(*slice)
    elif 'strain' in quantity and start_time < pd.Timestamp('2018-01-12',tz='Europe/Berlin'):
        file_info_temp = kwargs.get('file_info_temp',None)
        strain_start = slice[0]
        strain_end = slice[3]
        strain_fs = slice[4]
        logger.debug((strain_end-strain_start).total_seconds()*strain_fs)
        
        if file_info_temp is not None:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f): 
                # suppress printout from getting the temp slice to not confuse user when inspecting logs
                # if temp slice could not be read, a warning is printed 6 lines down
                temp_slice = get_slice(strain_start, strain_end-strain_start, quantity='temp', file_info=file_info_temp, upsample_fs=strain_fs)
        else:
            temp_slice = None
        
        if temp_slice is None:
            logger.warning("Can't read measurement file for temperature compensation {}:{}. Skipping!".format(start_time, quantity))
            return None
        
        Pt_ind = ['Pt100' in head for head in temp_slice[1]]
        Pt_mean = np.nanmean(temp_slice[5][:,Pt_ind], axis=1)
        for ind,head in enumerate(temp_slice[1]):
            if not 'Pt100' in head:
                continue
            if np.nanmax(temp_slice[5][:,ind])>80 or np.nanmin(temp_slice[5][:,ind])<-40:
                logger.info('Channel {} out of bounds. Max: {}, Min: {}'.format(head,np.nanmax(temp_slice[5][:,ind]), np.nanmin(temp_slice[5][:,ind])))
                temp_slice[5][:,ind]=Pt_mean

        #check out of bounds measurements
        slice = strain_manipulate_transform(*slice, quantity, temp_slice)
            
    start_time, headers, units, end_time, sample_rate, measurement = slice  
    
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
    
    slice = get_slice_corrected(start_time, duration, quantity, file_info, **kwargs)
    
    if slice is None:
        return None
    
    actual_start_time, headers, units, end_time, sample_rate, measurement  = slice
    
    if np.isnan(measurement).any():
        logger.warning("Measurement slice {}: {} contains nan. Skipping!".format(start_time, quantity))
        return None
    
    nyq = sample_rate/2
    dec_fact = round(sample_rate/target_fs)


    if 'strain' in quantity:
        channel_selector = np.array([False for header in headers])
        strain_channels = ['A_z','A_t','A_zt','B_z','B_t',
                                   'B_zt','C_z','C_t','C_zt','D_z','D_t','D_zt',]
        new_heads=[]
        for ij in range(len(headers)):
            if headers[ij] in strain_channels:
                channel_selector[ij]=True
                new_heads.append(headers[ij])
        measurement=measurement[:,channel_selector]
        headers=new_heads

    b, a = scipy.signal.butter(4, [highpass/nyq, lowpass/nyq], btype='band')
    
    ftype=scipy.signal.ltisys.dlti(b,a)
    meas_dec = scipy.signal.decimate(measurement, dec_fact, axis=0, ftype=ftype, zero_phase=True)
    
    measurement = meas_dec
    sample_rate = sample_rate/dec_fact      
    
    return actual_start_time, headers, units, end_time, sample_rate, measurement   

def wind_transform(file_time, headers, units, start_time, sample_rate, measurement):
    '''
    input: Wg, Wr
    output: Wg, Wr, Wx, Wy
    '''  
    
    new_headers = []
    new_meas = []
    new_units = []
    for pair in [('Wg','Wr'),('Wg_top', 'Wr_top')]:
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
            Wg=measurement[:,inds[0]]
            Wr=measurement[:,inds[1]]
            
            Wx,Wy = compensate_wind_jumps(Wr, Wg)
            
            Wr_,Wg = calc_ar(Wx, Wy)
            Wr_=np.degrees(Wr_)
            Wr_[Wr_<0]+=360
            
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


def compensate_wind_jumps(Wr, Wg):
    '''
    compensate for averaged jumps of wind direction between 0° and 360° caused by digitization of the wind signal at a lower sampling rate
    
    find main wind direction (alpha) by orthogonal least squares in cartesian coordinates
    transform a into range (alpha+-180°)
    while derivative of a contains values <-35 or >35
        substitute previous/following values of a forward and backward
        
    filter a in cartesian coords with a lowpass filter
        
    find main wind direction and compare with previous
    
    return a 
    '''

    def running_mean(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    Wr = np.copy(Wr)
    
    angle = orthogonal_lsq(azr= [Wr, Wg])

    Wr -=angle
    Wr[Wr>180] -= 360
    Wr[Wr<-180] += 360
    d_Wr=Wr[1:]-Wr[:-1]   
    
    d_allow = 30
    
    intp_inds = np.logical_or(d_Wr<-d_allow,d_Wr>d_allow)
    intp_inds = np.hstack((intp_inds,np.array([0])))
    intp_inds = np.array(intp_inds, dtype=bool)
        
    eps=0.00001
    std = 0
    new_std = np.std(d_Wr)
    counter=0    
    interpol_length = 1000
    overlap = 0.25

    while True:
        if new_std == 0:
            break
        if np.abs((new_std-std)/(new_std+std)/2)<eps:
            break
        
        counter+=1
        
        if counter >15:
            break

        step = int(interpol_length*(1-overlap))
        for i in range(0,len(Wr),step):
            imin,imax = (i - step,i + step)
            if imin<0:
                imin=0
            if imax> len(Wr):
                imax = len(Wr)
                
            this_intp_inds = intp_inds[imin:imax]
            this_t = np.linspace(0,len(this_intp_inds)-1, len(this_intp_inds))

            this_Wr = np.copy(Wr[imin:imax])

            ql,qml,qmu,qu = np.percentile(this_Wr, [10,40,60,90])

            if np.abs(qu-ql)>60:
                this_intp_inds = np.logical_or(np.logical_or(this_Wr>qu, this_Wr<ql), this_intp_inds)
            this_intp_inds = np.logical_and(np.logical_not(np.logical_and(this_Wr<qmu, this_Wr>qml)), this_intp_inds)

            knot_inds =np.logical_not(this_intp_inds)
            interp_y = this_Wr[knot_inds]
            interp_x = this_t[knot_inds]
                      
            interp_x_new = this_t[this_intp_inds]
            
            if not len(interp_x): 
                continue
            if len(interp_x_new)/len(interp_x)>1:
                continue

            k = min (len(interp_x),5)
            spl = scipy.interpolate.InterpolatedUnivariateSpline(interp_x, interp_y, k=5, ext=0, check_finite=False)

            interp_y_new = spl(interp_x_new)

            this_Wr[this_intp_inds] = interp_y_new         

            Wr[imin:imax] = this_Wr
  
        Wr +=angle
        angle = orthogonal_lsq(azr= [Wr, Wg])

        Wr -=angle
        Wr[Wr>180] -= 360
        Wr[Wr<-180] += 360

        d_Wr=np.diff(Wr)

        intp_inds = np.logical_or(d_Wr<-d_allow,d_Wr>d_allow)
        intp_inds = np.hstack((intp_inds,np.array([0])))
        intp_inds = np.array(intp_inds, dtype=bool)

        std=new_std
        new_std = np.std(d_Wr)

    Wr +=angle
    
    x,y = calc_xy(az=np.radians(Wr), r=Wg)

    x = running_mean(x,10)
    y = running_mean(y,10)

    return x,y

     
def calc_xy(az, r=1):
    x=r*np.cos(az) # for elevation angle defined from XY-plane up
    #x=r*np.sin(elev)*np.cos(az) # for elevation angle defined from Z-axis down
    y=r*np.sin(az) # for elevation angle defined from XY-plane up
    #y=r*np.sin(elev)*np.sin(az)# for elevation angle defined from Z-axis down
    return x,y

def calc_ar(x,y):
    xy = x**2 + y**2
    r = np.sqrt(xy)
    az = np.arctan2(y, x)
    return az, r  

def orthogonal_lsq(azr=None, xy=None, rad = False):
    
    assert azr is not None or xy is not None
    
    if azr is not None:
        assert len(azr) == 2
        az,r = azr
        if not rad:
            az = np.radians(az)
        x,y = calc_xy(az, r)
    else:
        x,y = xy
        az,r = calc_ar(x, y)
        
        
    vector = np.array([x,y]).T       
    U, S, V_T = np.linalg.svd(vector, full_matrices=False)
    
    angle = np.arctan2(-V_T[1,0],V_T[1,1])
    
    x_,y_ = calc_xy((az - angle),r)
    if np.sum(x_<0)>len(x_)/2:
        angle += np.pi
    if not rad:
        angle = np.degrees(angle)
    return angle

def strain_manipulate_transform(start_time, headers, units, end_time, sample_rate, measurement, quantity, temp_slice=None):
    
    measurement, deltas = fbg_strain_reader.manipulate_data(measurement, start_time, sample_rate, 
                                          previous_a=None, previous_delta=None, previous_start_time=None)
    
    # strain conversion
    
    # define strain channels 
    # per strain channel define temp_compensation channel -> may have to use qstation moni data (->f)
    # per strain channel define initial wavelength iwl/rwl/iw

    st=start_time
    
    S1 = 6.37E-06
    S2 = 7.46E-09
    k = 0.772
    alpha_steel = 12e-6
    alpha_glass = 0.55e-6

    #T=-S1/2/S2+np.sqrt((S1**2+4*S2*np.log(wl/iwl))/4/S2**2)+22.5
    if quantity == 'strain_rosettes':
        strain_t={
            'A_z':'A_Temp',  'A_t':'A_Temp',  'A_zt':'A_Temp',
            'B_z':'B_Temp', 'B_t':'B_Temp', 'B_zt':'B_Temp',
            'C_z':'C_Temp', 'C_t':'C_Temp', 'C_zt':'C_Temp',
            'D_z':'D_Temp_2', 'D_t':'D_Temp_2', 'D_zt':'D_Temp_2',
            }
    elif quantity == 'strain_bolts':
        strain_t={
            '10_z1':'10_Temp', '10_z2':'10_Temp',
            '8_z1':'8_Temp', '8_z2':'8_Temp', '8_z3':'8_Temp',
            '9_z1':'9_Temp', '9_z2':'9_Temp', '9_z3':'9_Temp',
            }
    else:
        raise RuntimeError
        
    
    # convert wl to strains and temperatures
    new_measurement = np.zeros_like(measurement)
    
    for ind, header in enumerate(headers):
        if header in config.temp_channels:
            
   # for temp_channel in config.temp_channels:
    #    try:
    #        ind = headers.index(temp_channel)
            T=-S1/2/S2+np.sqrt((S1**2+4*S2*np.log(measurement[:,ind]/config.initial_wl[header]))/4/S2**2)+22.5
            new_measurement[:,ind]=T
    #    except Exception as e:
    #        logger.warning(e)
    #        pass

        
    if temp_slice is not None:
        start_t, hds_t, _, dur_t, sample_rate_temp, meas_temp = temp_slice
        if quantity == 'strain_rosettes':
            temp_comp={
                'A_Temp'  :0.5*meas_temp[:,hds_t.index('Pt100_01')]+0.5*meas_temp[:,hds_t.index('Pt100_04')],
                'B_Temp'  :meas_temp[:,hds_t.index('Pt100_04')],
                'C_Temp'  :0.25*meas_temp[:,hds_t.index('Pt100_04')]+0.75*meas_temp[:,hds_t.index('Pt100_03')],
                'D_Temp_2':0.33*meas_temp[:,hds_t.index('Pt100_03')]+0.67*meas_temp[:,hds_t.index('Pt100_02')]}
        elif quantity == 'strain_bolts':
            temp_comp={
                '8_Temp'  :0.54*meas_temp[:,hds_t.index("Pt100_01")]+0.46*meas_temp[:,hds_t.index("Pt100_04")],
                '9_Temp'  :0.63*meas_temp[:,hds_t.index("Pt100_01")]+0.37**meas_temp[:,hds_t.index("Pt100_04")],
                '10_Temp' :0.72*meas_temp[:,hds_t.index("Pt100_01")]+0.28*meas_temp[:,hds_t.index("Pt100_04")]}
        
    else:
        temp_comp = {}
        for ind, header in enumerate(headers):
            if header in config.temp_channels:
#         for channel in config.temp_channels:
#             try:
#                 ind=headers.index(channel)
                temp_comp[header]=new_measurement[:,ind]
#             except:
#                 pass
        
    for channel,header in enumerate(headers):
        
        if header in config.temp_channels:
            pass
        else:
            comp_chan = strain_t[header]
            t = temp_comp[comp_chan]
            strain= 1/k*(np.log(measurement[:,channel]/config.initial_wl[header])-S1*(t-22.5)-S2*(t-22.5)**2)-(alpha_steel-alpha_glass)*(t-22.5)
            new_measurement[:,channel]=strain
            units[channel]='m/m'
            
    mean_=np.nanmean(new_measurement, axis=0)
    new_measurement_2 = new_measurement - mean_
    
    return start_time, headers, units, end_time, sample_rate, new_measurement

def get_modal_results(quantity: str, duration: pd.Timedelta, 
                      stats: xr.Dataset=None, create_new: bool=False, **kwargs):
    
    minutes = int(duration.total_seconds()/60)
    ds_path = os.path.join(config.db_root_path, f'{minutes}-minutes/', 'modal_{}.nc'.format(quantity))
    
    if not os.path.exists(ds_path):
        logger.warning(f'Path for modal does not exist {ds_path}.')

        
    if stats is None and create_new:
        raise RuntimeError('Statistics xarray has to be provided to run modal analysis.')
        
    if not create_new:
        stat_result = os.stat(ds_path, follow_symlinks=True)
        if config.ds_cache is not None and config.ds_cache[f'{quantity}_modal']['mtime']==stat_result.st_mtime:
            logger.info('Getting modal results for {} (cached)'.format(quantity))
            ds = config.ds_cache[f'{quantity}_modal']['ds']
        else:
            
            logger.info('Getting modal results for {}'.format(quantity))
            
            ds =  xr.open_dataset(ds_path, engine='h5netcdf')
            ds.load()
            ds.close()
            if config.ds_cache is not None:
                 config.ds_cache[f'{quantity}_modal']['ds'] =  ds
                 config.ds_cache[f'{quantity}_modal']['mtime'] = stat_result.st_mtime
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
    PreProcessSignals.load_measurement_file = dummy_load
    conf_dir = os.path.join(config.modal_conf_dir, quantity)
    
    if check_errors:
        stats = check_and_mark_errors(stats)
    
    minutes = int(duration.total_seconds()/60)
    
    process_ds_path = os.path.join(config.db_root_path, f'{minutes}-minutes/', 'modal_{}.{}.nc'.format(quantity,config.pid))
    master_ds_path = os.path.join(config.db_root_path, f'{minutes}-minutes/','modal_{}.nc'.format(quantity))
    
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
        
        #if quantity == 'accel' and start_time> pd.Timestamp('2017-11-01 00:00', tz = 'Europe/Berlin'):
            #Accel_01 failed silently and current statistical methods don't mark it as error
        #    continue
        
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
                slice = get_slice_preprocessed(start_time, duration, quantity, **kwargs)
        except Exception as e:
            logger.warning('Exception while trying to get preprocessed slice: ')
            logger.warning(e)
            slice=None
            
        if slice is None:
            this_ds.coords['time'] = [start_time.to_datetime64()]
        else:
            actual_start_time, headers, units, end_time, sample_rate, measurement = slice
            
    
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
                except:
                    logger.warning('error')
                    
            this_ds.coords['channels'] = np.array(headers,dtype=str)
            this_ds.coords['time'] = [start_time.to_datetime64()]
            
            try:
                #remove errorneous channels
                #if this_ds['error'].any():
                #    pass
                
                now=time.time()
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f): 
                    results = modal_analysis_single(start_time,slice, quantity, 
                                    duration)
                n, f, std_f, d, std_d, MPC, MP, MPD, MC, msh, s_vals_psd =results
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
    else:    
        # process_ds should not have changed on disk during processing    
        process_ds = save_ds(process_ds, process_ds, process_ds_path, what='modal')
        
        # master_ds almost certainly has changed on disk if multiple workers were processing files
        master_ds = save_ds(process_ds, master_ds, master_ds_path, reload_current = True, what='modal')
    
        # os.rename(process_ds_path, process_ds_path+'.old')
            
        logger.debug(f'Removing temporary dataset at {process_ds_path}')
        os.remove(process_ds_path)
        
    return master_ds

def modal_analysis_single(start_time, slice,  quantity, duration):
    
    conf_dir = os.path.join(config.modal_conf_dir, quantity)
    st=start_time
    
    skip_existing = True
    save_results = True
    interactive = False
    
    minutes = int(duration.total_seconds()/60)
    
    result_folder = os.path.join(config.slice_root_path, 
                                 f'{minutes}-minutes',
                                 'modal_{}'.format(quantity),
                                 '{}'.format(st.year),
                                 '{:02d}'.format(st.month))
        
    nodes_file = os.path.join(conf_dir, 'nodes')
    lines_file =os.path.join(conf_dir, 'lines')
    master_slaves_file=os.path.join(conf_dir, 'master_slaves')

    geometry_data = GeometryProcessor.load_geometry(nodes_file, lines_file, master_slaves_file)    
    
    conf_file = os.path.join(conf_dir, 'setup_info')
    chan_dofs_file = os.path.join(conf_dir, 'channel_dofs')
    ssi_file = os.path.join(conf_dir, 'ssi_config')      
    plscf_file = os.path.join(conf_dir, 'plscf_config')    
    
    fname = os.path.join(result_folder,
        '{:02d}_{:02d}-{:02d}_{:02d}-{:02d}_prep_data.npz'.format(
            st.year, st.month, st.day, st.hour, st.minute))
    
    if not os.path.exists(fname) or not skip_existing:
        start_time, headers, units, end_time, sample_rate, measurement \
            = slice
        
        accel_channels = []
        disp_channels = []
        if quantity == 'accel':
            ref_channels = []
            ref_channels.append(headers.index('Accel_01'))
            ref_channels.append(headers.index('Accel_02'))
            accel_channels = list(range(len(headers)))
            disp_channels = []
            chan_dofs_dict = {'Accel_01':     [1, 90,  0],
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
                              'Accel_04_top': [2, 0,   0]} 
            
            
        elif quantity == 'strain_rosettes':
            ref_channels = []
            ref_channels.append(headers.index('A_zt'))
            ref_channels.append(headers.index('B_zt'))  
            ref_channels.append(headers.index('C_zt'))
            ref_channels.append(headers.index('D_zt'))
            accel_channels = []
            disp_channels = list(range(len(headers)))
            chan_dofs_dict = {'A_z' : ['A_z_1',  0, 90],
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
                              'D_zt': ['D_zt_1', 0, 45]}
            
        prep_data = PreProcessSignals(measurement, sample_rate, 
                        ref_channels=ref_channels, accel_channels=accel_channels,
                        disp_channels= disp_channels, channel_headers=headers,
                        start_time=start_time)
        
        chan_dofs = [[chan]+chan_dofs_dict[header] for chan, header in enumerate(headers)]
        prep_data.add_chan_dofs(chan_dofs)
        s_vals_psd = prep_data.sv_psd(1444, method='blackman-tukey', refs_only=False)
        freqs = prep_data.freqs
        if save_results: 
            prep_data.save_state(fname)
    else:                                
        prep_data = PreProcessSignals.load_state(fname)
        s_vals_psd = prep_data.sv_psd(1444, method='blackman-tukey', refs_only=False)
        freqs = prep_data.freqs

    
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
        stabil_calc = StabilCluster(modal_data,prep_data)    
        
    else:
        try:
            stabil_calc= StabilCluster.load_state(fname, modal_data)
        except Exception as e:
            logger.warning(e)
            stabil_calc = StabilCluster(modal_data,prep_data)   
            
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
    modal_sorted = xr.Dataset()
    
    for modepair in range(5):
        f_range = [(0.34,0.38),(0.6,0.65),(1.2,1.4),(2.0,2.15),(3.2,3.55)][modepair]
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
    db_path = os.path.join(config.db_root_path, f'{minutes}-minutes/')
    
    if len(sys.argv) > 4: q_selector=int(sys.argv[4])
    else: q_selector = 1
    
    quantities = [
        'accel', #0
        'wind', #1
        'temp',#2
        'strain_rosettes',#3 
        #'strain_bolts'    
        ][q_selector:q_selector+1]

    for quantity in quantities:
    
        
        origin = config.origins[quantity]
        subpath = config.subpaths[origin]
        logger.info('Quantity: {}, Duration: {}'.format(quantity, minutes))
        
        if 0:
            file_contents = read_file(os.path.join(config.file_root_path, subpath, 'Wind_kontinuierlich__1_2018-06-13_15-00-00_000000.csv.bz2'))
            logger.debug(file_contents)
            meas = file_contents[-1]
            logger.debug(meas.shape)
            ts = meas[:,0]
            logger.debug(ts[-1]-ts[0])
            for i in range(meas.shape[0]):
                print(ts[i+1]-ts[i])
                if (ts[i+1]-ts[i])!=1:
                    print(i)
                    break
            return
        
        if 0: # create
            file_info = get_file_info(origin, create_new=True, skip_existing=True, reduced=False)
            return
        else: # get    
            file_info = get_file_info(origin, create_new=False)
        
        
        if 0:
            slice = get_slice_corrected(pd.Timestamp('2018-02-27 20:00',tz='Europe/Berlin'), 
                                        duration, 
                                        quantity, 
                                        file_info, 
                                        file_info_temp = get_file_info(config.origins['temp']))
            print(slice[:-1])
            print(slice[-1].shape)
            print(np.mean(slice[-1], axis=0))
            actual_start_time, headers, units, end_time, sample_rate, measurement = slice
            
            this_dict = describe_stats(measurement, headers, quantity)
            print(this_dict)
            return
        
        if 1:
            if 'strain' in quantity:
                file_info_temp = get_file_info(config.origins['temp'])
                
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
            modal = get_modal_results(quantity, duration, stats, 
                                      skip_existing=True, create_new=True, filter_errors=False, 
                                      chunksize=20, num_workers=num_workers, this_worker=this_worker)
            return
        

    return
    
                    
if __name__ == '__main__':
    main()
