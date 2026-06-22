
# coding: utf-8

import os
import warnings
import logging

import pytz
import datetime
# import dateutil
import time

berlin_dst=pytz.timezone('Europe/Berlin') # cet/cest

import numpy as np
import scipy.stats
import scipy.signal
import scipy.signal.ltisys

import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.transforms as tx
import matplotlib.dates
from matplotlib.ticker import LinearLocator
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter
import matplotlib.ticker
import locale

import pandas as pd
import xarray as xr
pd.plotting.register_matplotlib_converters()

import config

from main_v2 import get_file_info, get_stats, get_modal_results,\
    check_and_mark_errors, read_file, get_slice_corrected, get_slice_preprocessed

#global logger
logger = logging.getLogger(__name__)


def inspect_time_shifts():
    '''
     There are two measurement systems running independently:
         - Gantner Q.Station 101T Controller (acceleration, wind and temperature measurements)
         - Windows 7 PC (strain measurements)
    Throughout the operating period time was not synchronized properly due to many reasons
        - until 2018-01-... the Q.Station used it's own time, which was never properly synchronized, timezone was probably UTC
        - until 2018-05? the pc was set up to automatically switch to daylight saving, timezone was Europe/Berlin
        - the clocks of the Q.Station and the PC showed a slight relative drift, resulting in increasing time differences
        - in normal operation, the q.station would save the completed measurement file via samba onto the pc, otherwise files would be saved to the internal storage / usb storage
        - the strain measurement were saved directly to the pc's harddisk
    Every measurement file from both devices contains a timestamp representing the local devices starting time of the measurement
    It also has a file timestamp representing the storage devices end time of the measurement, i.e. when the last bit was written to the file
    This timestamp is usually preserved when the file is compressed, transfered to the server and decompressed
    But, on 2018-05-24 the script for compression was changed due to performance reasons, when handling the large amount of files, that have been accumulated during the the four years of operation.
    This resulted in the loss of the file timestamp of any file transfered since then.
    an error in the function "read_file" that was meant to correct the aforementioned issue causes inconsistent filetimes since then
    
    In order to synchronize the measurements created by both systems consistently throughout the full operating period we have to:
    - define a common time zone
    - correct daylight saving periods
    (- correct clock drifts)
    - derive a common start time from the local device's timestamp, the storage device's file timestamp and the duration of the measurement
    - make sure no artificial gaps are introduced by calculating new starting times
    - keep real gaps between files (e.g. due to reconfiguration, restarts, failures, etc)
    - this should  result in a synchronization accuracy of less than a minute
    
    In this framework file_time was used as a reference time troughout all functions, however it is not a reliable reference
    
    TODO: identify the different periods with respect to time settings by plotting the file timestamp versus the start time as recorded by the device    
    TODO: a function should be introduced, that creates a common start time for each file based on the given information, which will then be used as a reference
    TODO: all function that use the file_time array must be corrected accordingly
    '''
    def onpick(event):
        #event._xorig
        time_data = event.artist._xorig
        time_picked = np.datetime64(matplotlib.dates.num2date(event.mouseevent.xdata))
        time_ind = np.argmin(np.abs(time_data - time_picked))
        picked_time = event.artist._xorig[time_ind]
        logger.debug(ds.sel(time=picked_time)['file_name'])
        logger.debug(event)
    global ds
    origins = {'accel':'accel', 
              'wind':'wind', 
              'temp':'temp', 
              'strain_rosettes':'strain', 
              'strain_bolts':'strain'}
    
    fig1=plt.figure(1)
    fig1.canvas.mpl_connect('pick_event',onpick)
    for quantity in ['accel','strain_rosettes']:
        origin=origins[quantity]

        ds = get_file_info(origin)
        
        plt.figure(1)
        duration = ds['duration'].values.astype('int64')*1e-9
        device_start_time = ds['start_time']
        file_end_time = ds['file_time']
        file_start_time = file_end_time -duration
        
        ((device_start_time-file_start_time).astype('float64')/3600).plot(label='device_start_time - file_start_time_{}'.format(origin),ls='none',marker='.',markersize=1, picker=3)
        plt.title('"Device Start Time" minus "File Creation Time" [h]')
        plt.legend(loc='lower left')
        plt.xlabel('synchronized time')
    # last clock change happened on 2017-10-29 03:00 to 01:00
    for timestamp in pytz.timezone('Europe/Berlin')._utc_transition_times[100:106]:
        plt.axvline(timestamp, color='red', alpha=.5)
        plt.text(timestamp, 4, 'clock change', rotation=90)
        
        
    other_events={datetime.datetime(2018,1,15): 'illumisense was set up, 24 h long files',
                                datetime.datetime(2018,5,24): 'qstation compression script changed', #, file time lost
                                datetime.datetime(2019,4,30): ' pc was set to utc',
                                datetime.datetime(2018,12,10): 'controller started breaking',
                                datetime.datetime(2018,1,25): 'strain compression script changed, NTP was set-up', #, file time lost
                                datetime.datetime(2016,12,15):'Strain recordings were started', # synchronization is needed
                                }
    for event,s in other_events.items():
        plt.axvline(event, color='green', alpha=.5)
        plt.text(event, 4, s, rotation=90)
        
    plt.ylim((-4,4))
    plt.show() 
           
def plot_file_info(origin: str, check_errors: bool=True, filter_errors: bool=False):
    
    
    def onpick(event):
        #event._xorig
        time_data = event.artist._xorig
        time_picked = np.datetime64(matplotlib.dates.num2date(event.mouseevent.xdata))
        time_ind = np.argmin(np.abs(time_data - time_picked))
        picked_time = event.artist._xorig[time_ind]
        filename = ds.sel(time=picked_time)['file_name'].data.item()
        yn = input('read and plot {} (y/n)'.format(filename))
        if yn == 'y':
            filepath = os.path.join(config.file_root_path, config.subpaths[origin], filename)
            file_contents = read_file(filepath)
            if file_contents is None:
                logger.warning('File unreadable: {}'.format(filepath))
                return
            file_time, file_size, headers, units, start_time, sample_rate,measurement = file_contents
            
            plot_file(file_time, headers, units, start_time, sample_rate, measurement)
            
    global ds

    ds =  get_file_info(origin)
    logger.debug(ds.channels)
    #return
    ax = plt.subplot()
    i=0
    for channel in ds.channels:
        if 'Time' in channel.data.item():
            continue
        elif 'Tagessekunden' in channel.data.item():
            continue
        else:
            logger.debug(channel.values)
        data=ds.sel(channels=channel).get('mean')
        selector = xr.ufuncs.logical_and(~data.isnull(), data!=0)
        selector = selector * (i+1)
        selector.plot(marker='.',label=channel.values, ax=ax, ls='none')
        i+=1
    plt.xlim((datetime.datetime(2015,4,1), datetime.datetime.today()))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('')
    plt.show()

    for channel in ds.channels:
        this_range = config.ranges.get(channel.variable.item(),None)
#         if 'Accel' in channel.data.item():
#             continue
        if 'Time' in channel.data.item():
            continue
        elif 'Tagessekunden' in channel.data.item():
            continue
        else:
            logger.debug(channel.values)
        logger.info(channel, )
        fig,axes = plt.subplots(nrows=3, ncols=2, sharex='col',sharey='row', gridspec_kw={'width_ratios':[5,1]})
        fig.canvas.mpl_connect('pick_event',onpick)

        if check_errors:
            ds = check_and_mark_errors(ds, new=False)
              
        error=ds.sel(channels=channel).get('error')
        
        for num,group in [(0,['mean','min','max']),(1,['q05','q50','q95']),(2,[
            #'std',
            #'skewness',
            'kurtosis'
            ])]:
            for variable in group:
                logger.debug(num,variable)
                data=ds.sel(channels=channel).get(variable,None)
                
                if filter_errors:
                    data=data[error<1]
                if not len(data): continue
                
                data.plot(marker='.',label=variable, ax=axes[num,0], ls='none', picker=5, )
                
                
                #xr.DataArray().plot.    
                if this_range is not None:
                    data.plot.hist(bins=50, range= this_range, label=variable, ax=axes[num,1], density=True, orientation='horizontal')
                else:
                    data.plot.hist(bins=50, label=variable, ax=axes[num,1], density=True, orientation='horizontal')
            
        x = error[error>=1].time.variable.data
        logger.debug(x)
        for ax in axes[:,0]:            
            trans = tx.blended_transform_factory(ax.transData, ax.transAxes)
            
            ax.plot(np.repeat(x,3), np.tile([0,1, np.nan], len(x)), linewidth=1, color='r', alpha=.25, transform=trans)
            #plt.axvline()
        axes[0,0].legend()
        if this_range is not None:
            axes[0,0].set_ylim(this_range)
        axes[1,0].legend()
        if this_range is not None: axes[1,0].set_ylim(this_range)
        axes[2,0].legend()
        axes[2,0].set_ylim(-10,10)
        logger.debug(channel)
        plt.show()
        
def plot_file(file_time, headers, units, start_time, sample_rate, measurement):
    if 0:
        num_channels = len(headers)
        # figure size = 1x1.61, subplot sizes = 1x0.81
        # twice the number of subplots in columns than in rows
        nrows = int(np.ceil(np.sqrt(num_channels/2)))
        ncols = int(np.ceil(num_channels/nrows))
        logger.debug(sample_rate)
        time_=[file_time+datetime.timedelta(seconds=i/sample_rate) for i in range(measurement.shape[0])]
        logger.debug(time_)
        fig,axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,sharey=False)
        for row in range(nrows):
            for col in range(ncols):
                if row*ncols+col >= num_channels:
                    continue
                axes[row,col].plot(time_,measurement[:,row*ncols+col], color='blue',alpha=.3,marker=',',ls='solid',markersize=2,label=headers[row*ncols+col])
                logger.debug(np.sqrt(np.mean(np.square(measurement[:,row*ncols+col])))*1e6)
                axes[row,col].legend()
        fig.autofmt_xdate()
    else:
    
        
        num_channels = len(headers)
        
        logger.debug(sample_rate)
        time_=[file_time+datetime.timedelta(seconds=i/sample_rate) for i in range(measurement.shape[0])]
        logger.debug(time_)
        plt.figure()
        ax = plt.subplot()
        for channel in range(num_channels):
            ax.plot(time_,measurement[:,channel]-np.mean(measurement[:,channel]), alpha=.3,marker=',',ls='solid',markersize=2,label=headers[channel].replace('_','\_'))
            #ax.plot(time_,measurement[:,channel], alpha=.3,marker=',',ls='solid',markersize=2,label=headers[channel].replace('_','\_'))
            #ax.legend(loc='upper right')
        #fig.autofmt_xdate()
        #locator = matplotlib.dates.MinuteLocator(byminute=5)
        #ax.xaxis.set_major_locator(locator)
        myFmt = matplotlib.dates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(myFmt)
        #ax.set_ylabel('Acceleration [\si{\metre\per\second\squared}]')
        ax.set_ylabel('Strain [\si{\micro\metre\per\metre}]')
        
    plt.show()
    
def plot_waterfall(quantity: str, duration: pd.Timedelta, dtstart: np.datetime64):
    from pyOMA.core.PreProcessingTools import PreProcessSignals
    ds = get_modal_results(quantity, duration)
    minutes = int(duration.total_seconds()/60)
    
    ds = ds.isel(time=ds.time>dtstart)
    ds = ds.dropna(dim='channels', how='all')
    plt.figure(tight_layout=True)
    Sxx = []
    time_iterator = ds.time.values
    for start_time in time_iterator:
        
        st = pd.Timestamp(start_time, tz = 'Europe/Berlin')
        result_folder = os.path.join(config.slice_root_path, 
                             f'{minutes}-minutes',
                             'modal_{}'.format(quantity),
                             '{}'.format(st.year),
                             '{:02d}'.format(st.month))
        fname = os.path.join(result_folder,
        '{:02d}_{:02d}-{:02d}_{:02d}-{:02d}_prep_data.npz'.format(
            st.year, st.month, st.day, st.hour, st.minute))
        if not os.path.exists(fname): 
            # would fail, if the first file is missing
            Sxx.append(np.full_like(s_vals_psd, np.nan))
            freqs = prep_data.freqs
            continue
        prep_data = PreProcessSignals.load_state(fname)
        s_vals_psd = prep_data.sv_psd()[0,:]
        s_vals_psd = 10 * np.log10(np.abs(s_vals_psd))
        Sxx.append(s_vals_psd)
        freqs = prep_data.freqs
    Sxx = np.vstack(Sxx)
    plt.pcolormesh(time_iterator, freqs, Sxx.T, shading='gouraud')#, norm=matplotlib.colors.LogNorm())
    plt.gcf().autofmt_xdate()
    cbar = plt.colorbar(label='$\sigma_1(|PSD|)$ [dB]')
    # cbar.set_label('')
    plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    
    return plt.gcf()
    
        
    
def plot_daily(quantity: str, duration: pd.Timedelta, dtstart: np.datetime64):
    
    if quantity in ['accel', 'strain_rosettes']:
        modal = True
    else:
        modal = False
        
    if not modal:
        ds =  get_stats(quantity, duration)
    elif modal:
        ds = get_modal_results(quantity, duration)
    
    minutes = int(duration.total_seconds()/60)
    
    ds = ds.isel(time=ds.time>dtstart)
    # print(ds)
    ds = ds.dropna(dim='channels', how='all')
    #return
    n_channels = max(1, int(len(ds.channels))) # prevent failure on empty datasets
    # print(n_channels)
    
    fig, axs = plt.subplots(nrows=n_channels, figsize=(5.906, n_channels), sharex=True, tight_layout=True, dpi=300, squeeze=False)
    
    handles = []
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for i, channel in enumerate(ds.channels):
        
        # this_range = config.ranges.get(channel.variable.item(),None)
        ax = axs[i,0]
        
        data = ds.sel(channels=channel)
        
        l2d = ax.plot('time', 'mean', data=data, label=channel.data, color=colors[i])[0]
        handles.append(l2d)
        ax.fill_between(data['time'].data, data['min'].data, data['max'].data, data=data, alpha=0.5, color=colors[i])
        
        # ax.axhline(this_range[0])
        # ax.axhline(this_range[1])
    # axs[0,0].set_xlim((data.time.data.min(), data.time.data.max()))
    axs[0,0].set_xlim(xmin=dtstart)
    fig.legend(handles= handles, loc='center right')
    fig.autofmt_xdate()
    fig.suptitle(f"mean / min / max over {minutes} minutes for each channel of type {quantity}")
    
    if modal:
        y = ds['frequencies'].data
        x = ds['time'].data
        x = np.repeat(np.expand_dims(x, axis=1), repeats=y.shape[1], axis=1)
        plt.figure(figsize=(5.906,5.906/1.618), tight_layout=True, dpi=300)
        plt.scatter(x,y,marker='+',c='grey')
        plt.ylabel('Frequencies [Hz]')
        plt.yticks([0.35,0.62,1.31,2.06,3.36])
        plt.ylim((0,5))
        plt.grid(True, 'major','y', zorder=0, lw=0.1)
        plt.gcf().autofmt_xdate()
        fig2 = plt.gcf()
    else:
        fig2 = None
    return fig, fig2

def plot_stats(quantity: str, duration: pd.Timedelta, 
               check_errors: bool=True, filter_errors: bool=False, 
               modal: bool=False):
    
    def onpick(event):
        #event._xorig
        time_data = event.artist._xorig
        time_picked = np.datetime64(matplotlib.dates.num2date(event.mouseevent.xdata))
        time_ind = np.argmin(np.abs(time_data - time_picked))
        picked_time = event.artist._xorig[time_ind]
        ds2 = ds.sel(time=picked_time)
        yn = input('read and plot slice {}, duration {}? (y/n)'.format(pd.Timestamp(picked_time, tz='Europe/Berlin'), duration))
        if yn == 'y':
            if modal:
                slice=get_slice_preprocessed(pd.Timestamp(picked_time, tz='Europe/Berlin'), duration, quantity)
            else:
                slice=get_slice_corrected(pd.Timestamp(picked_time, tz='Europe/Berlin'), duration, quantity)
            plot_file(*slice)
            
    global ds
    
    if not modal:
        ds =  get_stats(quantity, duration)
    elif modal:
        ds = get_modal_results(quantity, duration)
    else:
        raise
    logger.debug(ds)
    #return
    

    for channel in ds.channels:
        #if channel != 'Accel_04':continue
        this_range = config.ranges.get(channel.variable.item(),None)
        logger.debug(channel, )
        fig,axes = plt.subplots(nrows=3, ncols=2, sharex='col',sharey='row', gridspec_kw={'width_ratios':[5,1]})
        fig.canvas.mpl_connect('pick_event',onpick)
        #fig=plt.figure()
        error=ds.sel(channels=channel).get('error')
        if check_errors:
            check_kurtosis = quantity in ['accel','strain_rosettes']
            check_and_mark_errors(ds, False, check_kurtosis)
            
        logger.debug(channel.channels.variable.data.item())
            
        for num,group in [(0,['mean','max-min']),(1,['q05','q50','q95']),(2,[
            #'var',
            #'skewness',
            'sample_rate'
            ])]:
            for variable in group:
                logger.debug(f'{num},{variable}')
                if variable == 'max-min':
                    data=ds.sel(channels=channel).get('max',None)-ds.sel(channels=channel).get('min',None)
                    data/=2
                else:
                    data=ds.sel(channels=channel).get(variable,None)
                logger.debug(len(data))
                
                
                if filter_errors:
                    data=data[error<1]
                if not len(data): continue
                if data.isnull().all():continue
                #data*=1e6
                if variable in ['max-min','rms']:
                    logger.debug('{}\t{}\t{:1.3f}'.format(channel.data, variable, data.max().data.item()))
                
                data.plot(marker='.',label=variable, ax=axes[num,0], ls='none', picker=5, )
                logger.debug(f'{variable},{data.median()}')
                
                continue

                if this_range is not None:
                    data.plot.hist(bins=50, range= this_range, label=variable, ax=axes[num,1], normed=True, orientation='horizontal')
                else:
                    data.plot.hist(bins=50, label=variable, ax=axes[num,1], normed=True, orientation='horizontal')
                    
        if check_errors:
            x = error[error>=1].time.variable.data
            logger.debug(x)
            for ax in axes[:,0]:            
                trans = tx.blended_transform_factory(ax.transData, ax.transAxes)
                
                ax.plot(np.repeat(x,3), np.tile([0,1, np.nan], len(x)), linewidth=1, color='r', alpha=.25, transform=trans)
                #plt.axvline()
        axes[0,0].legend()
        if this_range is not None:
            axes[0,0].set_ylim(this_range)
        axes[1,0].legend()
        if this_range is not None: axes[1,0].set_ylim(this_range)
        axes[2,0].legend()
        #axes[2,0].set_ylim(-2,5)
        logger.debug(channel)
        plt.show()
        
def load_filter_merge(quantity: str, duration: pd.Timedelta, 
                      time_range=None, 
                      kurt_range = None,
                      rms_range = None,
                      wind_range = 0,
                      temp_range = 0,
                      f_range = None,
                      damp_range = None,
                      mode_pair = 0,
                      mc_range = None, 
                      filter_errors=False, 
                       **kwargs):
    '''
    Loads:
        - modal result DataSet of selected quantity
        - statistics DataSet for wind
        - statistics DataSet for temperature
    for the selected block length of the signal
    
    Filters according to the given filter ranges. Ranges are all applied
    and-wise if this is not desired only provide one range at the time.
    
    Merges all DataSets into a single DataSet and returns it for further processing.
    
    Parameters:
    -----------
        quantity: str
            Quantity to load modal results from, must be one of 'accel' or 'strain_rosettes'
        duration: pd.Timedelta
            Define the block-length of the analysis must correspond to 10, 30, 60 or 120 minutes
        time_range: tuple (pd.Timestamp, pd.Timestamp), optional
            Defines the filter for DataSet coordinate 'time'
        kurt_range: tuple (lower_kurtosis, upper_kurtosis), optional
            Filters the kurtosis of the raw signal averaged over channels
        rms_range: tuple (lower_rms, upper_rms), optional
            Filters the DataSet based on the root-mean-square of the raw 
            signal averaged over channels
        wind_range: tuple (lower_windspeed, upper_windspeed) or integer, optional
            Filters the DataSet based on the measured windspeed of channel 'Wg'
            Alternatively an integer can be provided for predefined ranges:
            0: all speeds, 1: weak wind, 2: moderate wind, 3: strong wind
            
        temp_range: tuple (lower_temperature, upper_temperature) or integer, optional 
            Filter temperatures based on the median average over all 5 temperature channels
            Alternatively an integer can be provided for predefined ranges:
            0: (-20,40), 1: (-20,0), 2: (0,10), 3: (10,20), 4: (20,40)
        f_range: tuple (lower_frequency, upper_frequency), optional
            Filter identified frequencies. 
            Alternatively, mode_pair can be provided for predefined ranges. 
            f_range takes precedence over mode_pair
        damp_range: tuple (lower_damping_ratio, upper_damping_ratio), optional
            Filter identified damping ratios
        mode_pair: integer, optional
            Set f_range based on the following
            0: (0,4), 1: (0.33,0.37), 2: (0.57,0.65), 3: (1.07,1.47), 4: (1.98,2.16), 5:(3.2,3.5)
        mc_range: tuple (lower_modal_contribution, upper_modal_contribution), optional
            Filter non-physical modes, based on the modal contributions criterion
            Might not be available, depending on the OMA method, that was used for analysis
        filter_errors: bool, optional
            Filter entries that were marked as errorneous before, due to
            sensor overload, analysis errors, or others...
            

    
    
    relating modal_results to other quantities:
            f,   d, MC, MPC,MPD,order, ⟵ modal results
           _________________________
    time   | x | x | x | x | x | x |
    rms    | x | x | x |   |   |   |
    wind   | x | x | x |   |   |   |
    temp   | x | x | x |   |   |   |
    (kurt.)| x | x | x |   |   |   |
      ↑    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    related quantities
   
    relating modes with each other:
        MC vs. MC, msh vs msh (MAC)
   
    relating quantities with each other (strain vs. accel):
       f vs. f, d vs. d, MC vs. MC
    '''
    
    assert quantity in ['accel', 'strain_rosettes']
    assert isinstance(duration, pd.Timedelta)
    
    ds =  get_modal_results(quantity, duration)
    logger.debug(len(ds.time))
    
    wind_stats = get_stats('wind', duration)
    wind_stats.load()
    temp_stats = get_stats('temp', duration)
    temp_stats.load()
    
    ds, wind_stats, temp_stats = xr.align(ds, wind_stats, temp_stats, exclude=['channels','modes'])
    logger.debug(len(ds.time))
    if time_range is None:
        time_range = (pd.Timestamp('2015-05-20', tz='Europe/Berlin'), pd.Timestamp(ds.time.max().item(), tz='Europe/Berlin'))
    else:
        assert len(time_range)==2
        assert isinstance(time_range[0], pd.Timestamp)
        assert isinstance(time_range[1], pd.Timestamp)
    
    if isinstance(wind_range, int):
        #             ['all',     'weak wind','moderate wind','strong wind',custom scaling]
        wind_range = [(0,40),(0,5.4),(5.4,10.7),(10.7,25),(0,20)][wind_range]
    else:
        assert len(wind_range)==2
    
    #             [ 'all',    'frost'    , '0-10','10-20','20-...'] 
    if isinstance(temp_range,int):
        temp_range = [(-20,40),(-20,0),(0,10),(10,20),(20,40)][temp_range]
    else:
        assert len(temp_range)==2
    
    if f_range is not None:
        logger.warning(f"Using provided frequency ranges {f_range}, ignoring mode_pair {mode_pair}")
        assert len(f_range)==2
    else:
        #          ['all','first mode','second mode,'third mode', fourth mode', 'fifth mode']
        # f_range = [(0,4),(0.33,0.38),(0.57,0.65),(1.17,1.4),(2.0,2.15),(3.15,3.55)][mode]
        # f_range = [(0,4),(0.305,0.405),(0.57,0.675),(1.118,1.4975),(1.915,2.2),(3.13,3.69)][mode]
        f_range = [(0,4),(0.33,0.37),(0.57,0.65),(1.07,1.47),(1.98,2.16),(3.2,3.5)][mode_pair]
        
        
    if rms_range is None:
        if quantity=='accel':
            rms_range = (0,24)
        elif quantity=='strain_rosettes':
            rms_range = (0,70)
        logger.warning(f"Setting a pre-defined filter on RMS {rms_range}")
    else:
        assert len(rms_range)==2
            
    #f_range = None
    logger.info('wind: {}, temp: {}, mode: {}'.format(wind_range, temp_range, f_range))


    # pre-filtering:
    #    modal_results: frequency_range (always) i.e. mode, mode_assignment (if necessary)
    #    related_quantities: time_range, rms_range, wind_range, temperature_range, 
    
    # ranges affecting dimension: time
    if time_range is not None:
        logger.debug('Filter Time')
        ds = ds.sel(time=slice(*time_range))
    
    if filter_errors:
        logger.debug('Filter Errors')
        error_ind = (ds['min']==ds['max']).any(dim='channels')
        ds = ds.where(np.logical_not(error_ind))
    
    wind_arr = wind_stats['mean'].sel(channels='Wg').rename('wind') 
    if wind_range is not None:
        logger.debug('Filter Wind')
        wind_ind = np.logical_and(wind_arr>wind_range[0],wind_arr<=wind_range[1])
        ds = ds.where(wind_ind)
        wind_arr = wind_arr.where(wind_ind)
    ds = xr.merge([ds, wind_arr])
    
    temp_arr = temp_stats['q50'].mean(dim='channels').rename('temp')   
    #temp_arr = temp_stats['q50'].sel(channels='Pt100_01').rename('temp')    
    if temp_range is not None:
        logger.debug('Filter Temp')
        temp_ind = np.logical_and(temp_arr>temp_range[0],temp_arr<temp_range[1])
        ds = ds.where(temp_ind)
        temp_arr = temp_arr.where(temp_ind)
    ds = xr.merge([ds, temp_arr])
    
    # prepare averaged RMS DataArrays for filtering
    if quantity == 'accel':
        rms_arr = ds['rms'].sel(channels=['Accel_01','Accel_02']).mean(dim='channels').rename('rms_m')
        rms_arr*=1000
        kurt_arr = ds['kurtosis'].sel(channels=['Accel_01','Accel_02']).mean(dim='channels').rename('kurtosis_m')
    elif quantity == 'strain_rosettes':
        rms = ds['rms']
        rms = rms.sel(channels=['A_z','B_z', 'C_z', 'D_z'])
        rms = rms.mean(dim='channels')
        rms = rms.rename('rms_m')  
        rms_arr = ds['rms'].sel(channels=['A_z','B_z', 'C_z', 'D_z']).mean(dim='channels').rename('rms_m')   
        rms_arr*=10e6
        kurt_arr = ds['kurtosis'].sel(channels=['A_z','B_z', 'C_z', 'D_z']).mean(dim='channels').rename('kurtosis_m')

    ds = xr.merge([ds, rms_arr, kurt_arr])
    
    if rms_range is not None:
        logger.debug('Filter RMS')
        rms_ind = np.logical_and(rms_arr>=rms_range[0],rms_arr<=rms_range[1])
        ds = ds.where(rms_ind)
        
    if kurt_range is not None:
        logger.debug('Filter Kurtosis')
        kurt_ind = np.logical_and(kurt_arr>=kurt_range[0],kurt_arr<=kurt_range[1])
        ds = ds.where(kurt_ind)
   
    # drop all entries that are empty after filtering  
    ds = ds.dropna(dim='time', how='all')
    
    
    # ranges affecting dimension: modes
    modal_ind = ds['frequencies'].isnull()
    modal_ind = np.logical_not(modal_ind)
    if f_range is not None:
        f_ind = np.logical_and(ds['frequencies']>=f_range[0],ds['frequencies']<=f_range[1])
        modal_ind = np.logical_and(f_ind, modal_ind)
    if mc_range is not None:
        mc_ind = np.logical_and(ds['modal_contributions']>mc_range[0], ds['modal_contributions']<=mc_range[1])
        modal_ind = np.logical_and(modal_ind, mc_ind)
    if damp_range is not None:
        damp_ind = np.logical_and(ds['damping']>damp_range[0], ds['damping']<=damp_range[1])
        modal_ind = np.logical_and(modal_ind, damp_ind)

    # TODO: mode assignment would have to be done here
    
    # ranges affecting dimension: channels
    # None
    try:
        cv_f=ds['std_frequencies']/ds['frequencies']
        #cv_f=cv_f.where(cv_f<cv_f.quantile(0.95))
        # if not log_scale:
            # cv_f = xr.ufuncs.log10(cv_f)
        
        ds['cov_frequencies']=cv_f
        
        cv_d=ds['std_damping']/ds['damping']
        #cv_d=cv_d.where(cv_d<cv_d.quantile(0.95))
        # if not log_scale:
            # cv_d = xr.ufuncs.log10(cv_d)
        ds['cov_damping']=cv_d
        
        #data_quality_1 = ds['mean_svd_psd'].isel(channels=0,drop=True)/ds['energy_svd_psd'].isel(channels=0,drop=True)
        #data_quality_1 = ds['energy_svd_psd'].isel(channels=0,drop=True)
        data_quality_1 = ds['max_svd_psd'].isel(channels=0,drop=True)
        
        #data_quality_21 = ds['mean_svd_psd'].sel(channels='Accel_04_top',drop=True)/ds['energy_svd_psd'].sel(channels='Accel_04_top',drop=True)    
        #data_quality_21 = ds['energy_svd_psd'].sel(channels='Accel_04_top',drop=True)
        if quantity == 'accel':
            data_quality_21 = ds['mean_svd_psd'].sel(channels='Accel_04_top',drop=True)
            data_quality_21 =data_quality_21.rename('data_quality')
            
            #data_quality_22 = ds['mean_svd_psd'].sel(channels='Accel_06',drop=True)/ds['energy_svd_psd'].sel(channels='Accel_06',drop=True)
            #data_quality_22 = ds['energy_svd_psd'].sel(channels='Accel_06',drop=True)
            data_quality_22 = ds['mean_svd_psd'].sel(channels='Accel_06',drop=True)
            data_quality_22 =data_quality_22.rename('data_quality')
            
            data_quality_2 = xr.merge([data_quality_21,data_quality_22])
        else:
            data_quality_2 = ds['mean_svd_psd'].isel(channels=5,drop=True)
            data_quality_2 = data_quality_2.rename('data_quality')
            
        
        data_quality=data_quality_1/data_quality_2
        #data_quality.rename('data_quality')
        #data_quality = data_quality/ds['rms'].mean(dim='channels')
        #data_quality=data_quality.where(data_quality<data_quality.quantile(0.95))
        # if not log_scale:
            # data_quality = 10*xr.ufuncs.log10(data_quality)
        # else:
        data_quality = data_quality**10
        #data_quality = data_quality
        if quantity =='accel':
            ds = xr.merge([ds,data_quality])
        else:
            ds['data_quality']=data_quality
    except Exception as e:
        logger.exception(e)
        
    ds = ds.where(modal_ind)
    
    ds = ds.transpose('time','modes','channels')
    
    return ds, (time_range, rms_range, wind_range, temp_range, f_range, )
    
    
    
def postprocess_modal_results(quantity: str, duration: pd.Timedelta, 
                              time_range=None, kurt_range=None, rms_range=None, 
                              wind_range=None, temp_range=None, 
                              f_range=None, damp_range=None, mode_pair=0, mc_range=None, 
                              filter_errors=False, 
                              q_1=None, q_2=None, 
                              fig=None, axes=None, color='grey', hide_ticks=False, scatter=True, log_scale=False, **kwargs):
    '''
    relating modal_results to other quantities:
            f,   d, MC, MPC,MPD,order, ⟵ modal results
           _________________________
    time   | x | x | x | x | x | x |
    rms    | x | x | x |   |   |   |
    wind   | x | x | x |   |   |   |
    temp   | x | x | x |   |   |   |
    (kurt.)| x | x | x |   |   |   |
      ↑    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    related quantities
   
    relating modes with each other:
        MC vs. MC, msh vs msh (MAC)
   
    relating quantities with each other (strain vs. accel):
       f vs. f, d vs. d, MC vs. MC
    '''
    import io
    import pickle
    def open_in_new_window(e=None):
        logger.debug(e)
        if not isinstance(e, matplotlib.backend_bases.MouseEvent):
            return
        if not e.dblclick:
            return
        axes = e.inaxes
        inx = list(fig.axes).index(e.inaxes)
        buf = io.BytesIO()
        pickle.dump(fig, buf)
        buf.seek(0)
        fig2 = pickle.load(buf) 
        j=0
        for i, ax in enumerate(fig2.axes):
            if i != inx:
                fig2.delaxes(ax)
            else:
                axes=ax
                j=i
        num_rows = np.sqrt(len(fig2.axes)+1.25)+0.5
        if not j%num_rows:
            is_time=True
        else:
            is_time=False
    
        fig2.subplots_adjust(left=0.05)
        dummy = fig2.add_subplot(111)
        axes.set_position(dummy.get_position())
        dummy.remove()
        axes.xaxis.set_ticks_position('default')
        axes.xaxis.set_visible(True)
        if is_time:
            loc=matplotlib.dates.AutoDateLocator()
            axes.xaxis.set_major_locator(loc) 
            axes.xaxis.set_major_formatter(matplotlib.dates.AutoDateFormatter(loc))
        else: 
            axes.xaxis.set_major_locator(matplotlib.ticker.AutoLocator()) 
            #axes.xaxis.set_minor_locator(matplotlib.ticker.AutoLocator()) 
            axes.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()) 
        axes.xaxis.reset_ticks()
        
        axes.yaxis.set_ticks_position('default')
        axes.yaxis.set_visible(True)
        axes.yaxis.set_major_locator(matplotlib.ticker.AutoLocator()) 
        #axes.yaxis.set_minor_locator(matplotlib.ticker.AutoLocator()) 
        axes.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()) 
        axes.yaxis.reset_ticks()
        #axes.tick_params(axis='both', which='major', pad=15)
        fig2.show()
        
    # kernel density estimation
    if 0:
        import seaborn as sns
    else:
        sns = None
    
    ds, (time_range, rms_range, wind_range, temp_range, f_range, ) = \
         load_filter_merge(quantity, duration, 
                           time_range, kurt_range, rms_range, 
                           wind_range, temp_range, 
                           f_range, damp_range, mode_pair, mc_range, 
                           filter_errors)

    
    # channels={
        # '1': [[ 90, 180],['Accel_01', 'Accel_02']],
        # '4': [[180, 90 ],['Accel_03', 'Accel_04']],
        # '5': [[180, 90 ],['Accel_05', 'Accel_06']],
        # '6': [[180, 90 ],['Accel_07', 'Accel_08']],
        # '3': [[270, 180],['Accel_01_top', 'Accel_02_top']],       
        # '2': [[270, 0  ],['Accel_03_top', 'Accel_04_top']]}['1'][1]
        #
    # if 'dirs' in ds:
        # directions = np.degrees(ds['dirs'].sel(channels=channels[0], drop=True)).rename('directions')
        # ds['directions'] = directions
    
    modal_results=['frequencies',
                   'damping', 
                   #'cov_frequencies',
                   #'cov_damping',
                   #'modal_contributions', 
                   #'MPC',
                   #'MPD',
                   #'model_orders',
                   #'directions'
                   ]
    
    related_quantities = ['time',
                          'rms_m',
                          'wind',
                          'temp',
                          #'kurtosis_m',
                          #'data_quality',
                          ]
    if q_1 is None:
        all_keys_1 = related_quantities[1:]+modal_results
    else:
        assert isinstance(q_1, list)
        all_keys_1 = q_1
        
    if q_2 is None:
        all_keys_2 = related_quantities+modal_results
    else:
        assert isinstance(q_2, list)
        all_keys_2 = q_2
        
    if fig is None and axes is None:
        fig, axes = subplots(len(all_keys_1), len(all_keys_2), sharex='col', sharey='row', gridspec_kw={'hspace':0.05, 'wspace':0.05})
    single_ax = False
    if not isinstance(axes, np.ndarray):
        single_ax = True
        axes = np.array([[axes]])
    
        
    cid = fig.canvas.mpl_connect('button_press_event', open_in_new_window)
    
    if not single_ax:
        for ax in axes.flat:
            # Hide all ticks and labels
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    
            # Set up ticks only on one side for the "edge" subplots...
            if ax.is_first_col():
                ax.yaxis.set_ticks_position('left')
            if ax.is_last_col():
                ax.yaxis.set_ticks_position('right')
            if ax.is_first_row():
                ax.xaxis.set_ticks_position('top')
            if ax.is_last_row():
                ax.xaxis.set_ticks_position('bottom')
                
    q_strings={'time':'$t$',
               'rms_m':'RMS',
               'wind':'$v_w$ ',
               'temp':'$T$ ',
               'kurtosis_m':'$\kappa$ ',
               'frequencies':'$f$',
               'damping':'$\zeta$ ',
               'modal_contributions':'$\delta$',
               'cov_frequencies':'$c_{v_f}$',
               'cov_damping':'$c_{v_{\zeta}}$',
               'data_quality':'$\Delta_{\sigma_{PSD}}$',
               'directions':'$\circ$' }
            
    q_strings_units={'time':'$t$',
               'rms_m':'$\sigma$ [\si{\micro\metre\per\metre}]',
               'wind':'$v_w$ [\si{\metre\per\second}]',
               'temp':'$T$ [\si{\celsius}]',
               'kurtosis_m':'$\kappa$ ',
               'frequencies':'$f$ [\si{\hertz}]',
               'damping':'$\zeta$ [\si{\percent}]',
               'modal_contributions':'$\delta$ [-]' }
    
    ds = ds.transpose('time','modes','channels')
    
    for row,q_1 in enumerate(all_keys_1):
        
        for col,q_2 in enumerate(all_keys_2):
            logger.debug('{} {}'.format(q_1,q_2))
            
            ax = axes[row,col]
            
            if single_ax:
                ax.set_ylabel(q_strings[q_1],rotation=0, labelpad=10, horizontalalignment='left')
            else:
                if ax.is_first_col() and not hide_ticks:
                    ax.yaxis.set_visible(True)
                if ax.is_last_col() or single_ax:
                    ax.yaxis.set_visible(True)
                    ax.set_ylabel(q_strings[q_1],rotation=0, labelpad=10, horizontalalignment='left')
                    ax.yaxis.set_label_position('right')
                    plt.setp(ax.get_yticklabels(), visible=False)
                    ax.tick_params(axis='y', which='both', length=0)
    
                if ax.is_last_row() or single_ax:
                    ax.xaxis.set_visible(True)
                    ax.set_xlabel(q_strings[q_2])
                    if hide_ticks:
                        ax.set_xticks([])
              
            # if q_1 in modal_results:
                # y = ds[q_1].where(modal_ind).data
            # else:
            y = ds[q_1].data
            logger.debug(y.shape)
            if np.issubdtype(y.dtype, np.datetime64):
                nan_func_2 = np.isnat
            else:
                nan_func_2 = np.isnan  
                
            if row+1==col:
                if not np.issubdtype(y.dtype, np.datetime64):
                    y = y.flatten()
                    ind = np.logical_not(nan_func_2(y))
                    y = y[ind]
                    
                    if sns is not None:
                        sns.distplot(y, ax=ax)
                    #elif 'cov' in q_1:
                    #    ax.hist(y, bins=100, 
                    #            #range=[*np.percentile(y, q=[1,99])],  
                    #            normed=True,
                    #            log=True,
                    #            color=color)                        
                    else:
                        ax.hist(y, bins=100, 
                                #range=[*np.percentile(y, q=[1,99])],  
                                density=True,
                                color=matplotlib.colors.to_rgba(color, alpha=0.8))
            else:
                
                # if q_2 in modal_results:
                    # x = ds[q_2].where(modal_ind).data
                # else:
                x = ds[q_2].data
                    
                if np.issubdtype(x.dtype, np.datetime64):
                    nan_func_1 = np.isnat
                else:
                    nan_func_1 = np.isnan   
 
                             
#                 if len(y.shape)==2 and len(x.shape)==1:
#                     
#                     naxis = int(ds[q_1].dims.index(ds[q_2].dims[0]))
#                     axis =  int(not naxis)
#                     x = np.repeat(np.expand_dims(x, axis=naxis), repeats=y.shape[axis], axis=naxis)                
#     
#                 if len(x.shape)==2 and len(y.shape)==1:
#                     
#                     naxis = int(ds[q_2].dims.index(ds[q_1].dims[0]))
#                     axis =  int(not naxis)
#                     
#                     y = np.repeat(np.expand_dims(y, axis=naxis), repeats=x.shape[axis], axis=naxis)                
                if len(y.shape)==2 and len(x.shape)==1:
                    x = np.repeat(np.expand_dims(x, axis=1), repeats=y.shape[1], axis=1)
    
                if len(x.shape)==2 and len(y.shape)==1:
                    y = np.repeat(np.expand_dims(y, axis=1), repeats=x.shape[1], axis=1)   
                y = y.flatten()
                # print(np.nanmean(y), np.nanstd(y))
                x = x.flatten()
 
                # remove nans

                ind = np.logical_or(nan_func_1(x), nan_func_2(y))
                ind = np.logical_not(ind)            
                y = y[ind]
                x = x[ind]
                
                if kwargs.get('predict', None) is not None:
                    # list of factors for powers of x (last is power 0)
                    y_ = np.zeros_like(y)
                    for i,factor in enumerate(reversed(kwargs['predict'])):
                        y_ += factor * y**i
                    y = y_
                if col>row+1:
                    if not (np.issubdtype(y.dtype, np.datetime64) or np.issubdtype(x.dtype, np.datetime64)):
                        
                        rho, pval = scipy.stats.spearmanr(x,y)
                        #corr_coef = np.corrcoef(x,y)
                        #rho = corr_coef[1,0]
                        #corr_coef = rho
                        
                        ax.annotate('${:1.3f}$'.format(rho), (0.5, 0.5), 
                                               xycoords='axes fraction',
                                               ha='center', va='center')
                else:
                    
                    #xscale='linear'
                    #yscale='linear'
                    #if 'cov' in q_2: 
                    #    xscale='log'
                    #if 'cov' in q_1: yscale='log'
                        # y = y.astype(np.int64)
                        
                    if scatter:# or (np.issubdtype(y.dtype, np.datetime64) or np.issubdtype(x.dtype, np.datetime64)):
                        ax.plot(x,y, ls='none', marker='.', markerfacecolor=color,markeredgecolor='none',markersize=1, 
                            alpha=0.75, 
                            zorder=0)
                        
                        if False: # linear regression
                            A = np.vstack([x, np.ones(len(x))]).T
                            curve_fit = np.linalg.lstsq(A, y, rcond=None)[0]
                            print(curve_fit)
                            xlin = np.array([x.min(), x.max()])
                            ylin = xlin*curve_fit[0]+curve_fit[1]
                            ax.plot(xlin,ylin)#, ls='none', marker='.', alpha=0.2, color='red', markersize=1, )
                    #    ax.set_xscale(xscale)
                    #    ax.set_yscale(yscale)        
                    else:

                        if np.issubdtype(x.dtype, np.datetime64):
                            x_ = matplotlib.dates.date2num(x)
                        else:
                            x_ = x
                        if np.issubdtype(y.dtype, np.datetime64):
                            y_ = matplotlib.dates.date2num(y)
                        else:
                            y_ = y
                        from matplotlib.colors import LinearSegmentedColormap
                        

                        color_s = matplotlib.colors.to_rgba(color, alpha=0)
                        # color_s = matplotlib.colors.to_rgba('white')
                        color_e = matplotlib.colors.to_rgba(color, alpha=1)
                        
                        
                        cmap = LinearSegmentedColormap.from_list('CustomCmap', colors=[color_s,color_e]) # fff white with alpha
                        if log_scale == 'x':
                            xscale='log'
                            yscale='linear'
                        elif log_scale == 'y':
                            yscale='log'
                            xscale='linear'
                        elif log_scale:
                            xscale,yscale = 'log','log' 
                        else:
                            xscale,yscale = 'linear','linear' 

                            
                        axes_size= (int(np.ceil(ax.bbox.width)), int(np.ceil(ax.bbox.height)))#3 in points
                        if xscale=='log':
                            xbins = np.logspace(np.log10(np.nanmin(x_)), np.log10(np.nanmax(x_)), axes_size[0])
                        else:
                            xbins=axes_size[0]
                        if yscale=='log':
                            ybins = np.logspace(np.log10(np.nanmin(y_)), np.log10(np.nanmax(y_)), axes_size[1])
                        else:
                            ybins=axes_size[1]
                        xbins //= kwargs.get('bin_factor',1)
                        ybins //= kwargs.get('bin_factor',1)
                        #hist2d produces artifacts when using alpha colormaps due to internal use of pcolormesh (draws Patches)
                        #ax.hist2d(x_, y_, bins=(xbins, ybins), density=True, cmap=cmap, norm=matplotlib.colors.LogNorm(), zorder=10)
                        
                        # compute the histogram and draw it as image
                        counts, xedges, yedges=np.histogram2d(x_,y_,(xbins, ybins), density=True)
                        ax.imshow(counts.T, cmap=cmap, norm=matplotlib.colors.LogNorm(), zorder=10,
                                   # interpolation='none',
                                   origin='lower', resample=False, aspect='auto',
                                   extent=(xedges.min(),xedges.max(), yedges.min(), yedges.max()))
                        
                    if sns is not None:
                        if not (np.issubdtype(x.dtype, np.datetime64) or np.issubdtype(y.dtype, np.datetime64)):
                            sns.kdeplot(x, y,  ax=ax, zorder=1)
                        
                
                
            if ax.is_last_row() or single_ax:
                if np.issubdtype(x.dtype, np.datetime64):
                    
                    ax.set_xlim(time_range)
                    rotation=30 
                    ha='right'
                    for label in ax.get_xticklabels():
                        #continue
                        label.set_ha(ha)
                        label.set_rotation(rotation)
                else:
                    if q_2 == 'damping':
                        ax.set_xlim(damp_range)
                    elif q_2 == 'modal_contributions':
                        ax.set_xlim((0,1))
                    elif q_2 == 'frequencies':
                        # plot_bounds = [(0,4),(0.30,0.40),(0.575,0.675),(1.25,1.35),(1.95,2.24),(3.13,3.69)][mode]
                        plot_bounds = f_range #[(0,4),(0.30,0.40),(0.575,0.675),(1.25,1.35),(1.95,2.24),(3.13,3.69)][mode]
                        ax.set_xlim(plot_bounds)
                    elif q_2 == 'wind':
                        ax.set_xlim(wind_range)
                    elif q_2 == 'temp':
                        ax.set_xlim(temp_range)
                    elif q_2 == 'rms' and rms_range is not None:
                        ax.set_xlim(rms_range)
                    else:
                        if len(x):
                            ax.set_xlim(*np.percentile(x, q=[0.01,99.99]))
                    if not hide_ticks:
                        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6, 
                                                                             prune='both'
                                                                             ))
            if ax.is_first_col() or single_ax:
                if q_1 == 'damping':
                    ax.set_ylim(damp_range)
                elif q_1 == 'modal_contributions':
                    ax.set_ylim((0,1))
                elif q_1 == 'frequencies':
                    # plot_bounds = [(0,4),(0.30,0.40),(0.575,0.675),(1.25,1.35),(1.95,2.24),(3.13,3.69)][mode]
                    plot_bounds = f_range
                    ax.set_ylim(plot_bounds)
                elif q_1 == 'wind':
                    ax.set_ylim(wind_range)
                elif q_1 == 'temp':
                    ax.set_ylim(temp_range)
                elif q_1 == 'rms' and rms_range is not None:
                    ax.set_ylim(rms_range)
                else:
                    if len(y) and not np.issubdtype(y.dtype, np.datetime64):
                        ax.set_ylim(*np.percentile(y, q=[0.01,99.99]))
                if not hide_ticks:
                    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6, 
                                                                             prune='both'
                                                                             ))

    # (0.343,0.368), (0.598,0.638), (1.236,1.373), (1.99,2.14), (3.154,3.557)
    # (0.340,0.372), (0.602,0.639), (1.239,1.355), (2.02,2.15),(3.21,3.548)
    #fig.autofmt_xdate()
    #plt.show()
    if not single_ax:
        plt.subplots_adjust(left=0.02, right=0.95, top=0.97, bottom=0.07)
        if not hide_ticks:
            plt.subplots_adjust(left=0.04, right=0.95, top=0.97, bottom=0.09)
    return
  
def modal_dirs(quantity, duration, update=False, plot_=False):
    
    assert quantity=='accel'
    assert isinstance(duration, pd.Timedelta)
    warnings.warn("Modeshapes are mostly missing from result db, would have to reread them all from disk.")
    modal = get_modal_results(quantity, duration)
    chan_dofs={
        '1': [[ 90, 180],['Accel_01', 'Accel_02']],
        '4': [[180, 90 ],['Accel_03', 'Accel_04']],
        '5': [[180, 90 ],['Accel_05', 'Accel_06']],
        '6': [[180, 90 ],['Accel_07', 'Accel_08']],
        '3': [[270, 180],['Accel_01_top', 'Accel_02_top']],       
        '2': [[270, 0  ],['Accel_03_top', 'Accel_04_top']]}
    
    if update:   
        dirs = np.empty((modal['modeshapes'].data.shape))
        dirs[:,:,:]=np.nan

        for node in ['1','2','3','4','5','6']:
            this_az, this_heads = chan_dofs[node] 
            pos_indexers, new_indexes = xr.core.indexing.remap_label_indexers(modal, {'channels':this_heads})                
            msh = modal['modeshapes'].data[:,:,pos_indexers['channels']]
            if np.all(np.isnan(msh)): continue
            for j,az in enumerate(this_az):
                if az == 0: x=msh[:,:,j]
                elif az == 90: y=msh[:,:,j]
                elif az == 180: x=-1*msh[:,:,j]
                elif az == 270: y=-1*msh[:,:,j]
            
            ax=np.abs(x)
            ay=np.abs(y)
            
            phix=np.angle(x)
            phiy=np.angle(y)
            
            ax2=ax**2
            ay2=ay**2
            
            tmin =-0.5*np.arctan2((ax2*np.sin(2*phix+np.pi)+ay2*np.sin(2*phiy+np.pi)),(ax2*np.cos(2*phix+np.pi)+ay2*np.cos(2*phiy+np.pi)))
            tmax =-0.5*np.arctan2((ax2*np.sin(2*phix      )+ay2*np.sin(2*phiy      )),(ax2*np.cos(2*phix      )+ay2*np.cos(2*phiy      )))
            
            #alphamin=np.arctan2(ay*np.cos(tmin+phiy),ax*np.cos(tmin+phix))
            alphamax=np.arctan2(ay*np.cos(tmax+phiy),ax*np.cos(tmax+phix))
            
            rmin=np.sqrt((ay*np.cos(tmin+phiy))**2 + (ax*np.cos(tmin+phix))**2)
            rmax=np.sqrt((ay*np.cos(tmax+phiy))**2 + (ax*np.cos(tmax+phix))**2)
            
            alphamax[alphamax<0] += np.pi

            dirs[:,:,pos_indexers['channels'][0]]=alphamax
            dirs[:,:,pos_indexers['channels'][1]]=rmin/rmax
    
        modal['dirs']=(('time','modes','channels'),dirs)
        modal.to_netcdf(os.path.join(path,'result_db/modal_{}_dirs.nc'.format(quantity)), engine='h5netcdf')    
    

    def get_dirs_transformed(modal,chan_dofs, node, modepair):
        
        alpha=[None,None]
        rratio=[None,None]
        if node == '1':
            
            logger.debug(modal.time.astype('datetime64[ns]'))
            modal=modal.where(np.logical_or(modal.time<np.datetime64('2017-11-23'),modal.time>np.datetime64('2018-02-11')))
        #for mode, data_func in enumerate([data.frequencies.argmin, data.frequencies.argmax]):
        for mode in range(2):
            
            this_alpha = modal.sel(channels=chan_dofs[node][1][0])['dirs'].isel(modes=mode+2*modepair).data
            
            this_rratio = modal.sel(channels=chan_dofs[node][1][1])['dirs'].isel(modes=mode+2*modepair).data
            
            x = (1 - this_rratio) * np.cos(this_alpha)
            y = (1 - this_rratio) * np.sin(this_alpha)            
            vector = np.array([x,y]).T  
            vector = vector[~np.isnan(vector).any(axis=1),:]
            U, S, V_T = np.linalg.svd(vector, full_matrices=False)
            
            mean_alpha = np.degrees(np.arctan(-V_T[1,0]/V_T[1,1]))   
            this_alpha = np.degrees(this_alpha)
            if mean_alpha < 0  : mean_alpha += 180
            if mean_alpha > 180: mean_alpha -= 180
            if mean_alpha > 90 : this_alpha[this_alpha<mean_alpha-90] += 180
            if mean_alpha <= 90: this_alpha[this_alpha<mean_alpha+90] += 180
            logger.debug(mean_alpha)
            alpha[mode] = this_alpha
            rratio[mode] = this_rratio
            
        this_time = modal.time.data
        
        return this_time, alpha, rratio
    
    def plot_time(modal, chan_dofs, node='1'):
        
        colors = list(matplotlib.cm.hsv(np.linspace(0, 1, 10)))
        axes=[]
        
        for i in range(5):
            plt.figure(tight_layout=1)
            axes.append(plt.axes())

        for modepair in range(5):
            
            this_time, (alpha_1, alpha_2), (rratio_1,rratio_2) = get_dirs_transformed(modal, chan_dofs, node, modepair)
            
            c_1=np.tile(colors[modepair], [len(alpha_1),1])
            c_1[:,-1]=1-np.array(rratio_1)
            
            axes[modepair].scatter(this_time, alpha_1, c=c_1, marker='.', s=4, edgecolors='none',label='$\\alpha_{{{0}}}$'.format(modepair*2+1))
            
            c_2=np.tile(colors[modepair+5], [len(alpha_2),1])
            c_2[:,-1]=1-np.array(rratio_2)
            
            axes[modepair].scatter(this_time, alpha_2, c=c_2, marker='.', s=4, edgecolors='none',label='$\\alpha_{{{0}}}$'.format(modepair*2+2))
            
            axes[modepair].set_ylabel('Hauptrichtung der Eigenform')
            axes[modepair].set_yticks((0,45,90,135,180,225,270,315,360))
            axes[modepair].set_yticklabels(['N','','O','','S','','W','','N'])
            axes[modepair].set_ylim((0,360))
            #axes[modepair].set_xlim((min(xtime),max(xtime)))
            
            leg=axes[modepair].legend(loc=(0.025,0.85), markerscale=5)
            leg.get_frame().set_alpha(0.45)
            axes[modepair].figure.autofmt_xdate()
    
    def plot_histograms(modal, chan_dofs, node='1', modepair=1, ax1=None, ax2=None,orientation ='horizontal'):    
        colors = list(matplotlib.cm.hsv(np.linspace(0, 1, 10)))
        
        this_time, (alpha_1, alpha_2), (rratio_1,rratio_2) = get_dirs_transformed(modal, chan_dofs, node, modepair)
        
        
        if ax1 is None: 
            plt.figure()
            fig1=plt.gcf()
            ax1 = fig1.gca()
        else:
            fig1 = None
        ax1.hist(alpha_1,label='$\\alpha_{{{}}}$'.format(modepair*2+1),bins=180, range=(np.nanmin(alpha_1), np.nanmax(alpha_1)), alpha=0.5, ls=None, lw=0, weights=1-np.array(rratio_1), color=colors[modepair], orientation=orientation, zorder=5-modepair)
        ax1.hist(alpha_2,label='$\\alpha_{{{}}}$'.format(modepair*2+2),bins=180, range=(np.nanmin(alpha_2), np.nanmax(alpha_2)), alpha=0.5, ls=None, lw=0, weights=1-np.array(rratio_2), color=colors[modepair+5], orientation=orientation, zorder=5-modepair)
        if orientation=='horizontal':
            #ax1.axhline(angle1, color=colors[2*modepair], ls='dotted',lw=0.5)
            #ax1.axhline(angle2, color=colors[2*modepair+1], ls='dotted', lw=0.5)
            ax1.set_ylim((0,360))
            ax1.set_xlim((0,ax1.get_xlim()[1]*1.05))
        elif orientation=='vertical':
            #ax1.axvline(angle1, color=colors[2*modepair], ls='dotted',lw=0.5)
            #ax1.axvline(angle2, color=colors[2*modepair+1], ls='dotted', lw=0.5)
            ax1.set_xlim((0,360))
            ax1.set_ylim((0,ax1.get_ylim()[1]*1.05))
            ax1.set_yticks([]) 
            ax1.set_xticks([0,45,90,135,180,225,270,315,360])
            ax1.set_xticklabels(['N','','O','','S','','W','','N'])
            ax1.set_xlabel('Hauptrichtung der Eigenform')
            leg=ax1.legend()
            leg.get_frame().set_alpha(0.45)
        
        if ax2 is None: 
            plt.figure()
            fig2=plt.gcf()
            ax2 = fig2.gca()
        else:
            fig2 = None
        ax2.hist(rratio_1,label='r/r {}'.format(modepair*2+0),bins=50, range=(0,1), alpha=0.5, ls=None, lw=0)
        ax2.hist(rratio_2,label='r/r {}'.format(modepair*2+1),bins=50, range=(0,1), alpha=0.5, ls=None, lw=0)
        ax2.legend()
        ax2.set_xlim((0,1))
        ax2.set_ylim((0,plt.ylim()[1]))
        
    def plot_boxplots(modal, chan_dofs, modepair=0):
        
        dirs1=[]
        dirs2=[]
        rrats1=[]
        rrats2=[]
        
        for node in ['1','4','5','6','3','2']:
            
            this_time, (alpha_1, alpha_2), (rratio_1,rratio_2) = get_dirs_transformed(modal, chan_dofs, node, modepair)
            alpha_1 = alpha_1[rratio_1<0.25]
            alpha_2 = alpha_2[rratio_2<0.25]
            dirs1.append(alpha_1[~np.isnan(alpha_1)])
            dirs2.append(alpha_2[~np.isnan(alpha_2)])

        plt.figure(figsize=(5,3.13*2),tight_layout=True)
        bp = plt.gca().boxplot(dirs1, vert=False, patch_artist=True)
        
        ## change outline color, fill color and linewidth of the boxes
        for box in bp['boxes']:
            # change outline color
            box.set( color='#4E8DBF', linewidth=1)
            # change fill color
            box.set( facecolor = 'white' )
        
        ## change color and linewidth of the whiskers
        for whisker in bp['whiskers']:
            dashes = (0, (3, 1, 1, 1))
            whisker.set(color='#4E8DBF', linewidth=1, linestyle=dashes)
        
        ## change color and linewidth of the caps
        for cap in bp['caps']:
            cap.set(color='#4E8DBF', linewidth=1)
        
        ## change color and linewidth of the medians
        for median in bp['medians']:
            median.set(color='#E63334', linewidth=1)
        
        ## change the style of fliers and their fill
        for flier in bp['fliers']:
            flier.set(marker='.',markersize=2, color='#E63334', alpha=0.25)
        #plt.gca().violinplot(dirs1,showmedians=True,bw_method='scott',vert=False)
        plt.yticks([1,2,3,4,5,6],['108','126','145','160','188','TMD'])
        plt.xticks([0,45,90,135,180,225,270,315,360],['N','','O','','S','','W','','N'])
        plt.xlim((0,360))
        plt.ylabel('Messpunkt / H\\"ohe [\si{\metre}]')
        plt.xlabel('Hauptrichtung')
        plt.gca().get_xaxis().tick_bottom()
        plt.gca().get_yaxis().tick_left()
        fig1=plt.gcf()
        plt.figure(figsize=(5,3.13*2),tight_layout=True)
        bp = plt.gca().boxplot(dirs2, vert=False, patch_artist=True)
        ## change outline color, fill color and linewidth of the boxes
        for box in bp['boxes']:
            # change outline color
            box.set( color='#4E8DBF', linewidth=1)
            # change fill color
            box.set( facecolor = 'white' )
        
        ## change color and linewidth of the whiskers
        for whisker in bp['whiskers']:
            dashes = (0, (3, 1, 1, 1))
            whisker.set(color='#4E8DBF', linewidth=1, linestyle=dashes)
            #whisker.set_dashes(dashes)
        
        ## change color and linewidth of the caps
        for cap in bp['caps']:
            cap.set(color='#4E8DBF', linewidth=1)
        
        ## change color and linewidth of the medians
        for median in bp['medians']:
            median.set(color='#E63334', linewidth=1)
        
        ## change the style of fliers and their fill
        for flier in bp['fliers']:
            flier.set(marker='.',markersize=2, color='#E63334', alpha=0.25)
        #plt.gca().violinplot(dirs2,showmedians=True,bw_method='scott',vert=False)
        plt.yticks([1,2,3,4,5,6],['108','126','145','160','188','TMD'])
        plt.xticks([0,45,90,135,180,225,270,315,360],['N','','O','','S','','W','','N'])
        plt.xlim((0,360))
        plt.gca().get_xaxis().tick_bottom()
        plt.gca().get_yaxis().tick_left()
        plt.ylabel('Messpunkt / H\\"ohe [\si{\metre}]')
        plt.xlabel('Hauptrichtung')
        fig2=plt.gcf()
        
        plt.show()
            
    def dirs_wind(wind_ds, modal_sorted, chan_dofs, node='1', target='Wr_top', ax1=None, axes=[]):
        logger.debug(modal_sorted)
        
        wind_ds = wind_ds.sel(channels=target)
        wind_ds = wind_ds.where(wind_ds>0)
        
        modal_sorted, wind_ds = xr.align(modal_sorted, wind_ds, exclude=['channels','modes'])
        
        logger.debug(wind_ds.dropna(how='any',dim='time'))
        logger.debug(modal_sorted.dropna(how='any',dim='time'))
        import matplotlib.cm
        colors = list(matplotlib.cm.hsv(np.linspace(0, 1, 10)))
        if not axes:
            for i in range(2):
                plt.figure(tight_layout=1)
                axes.append(plt.axes())   
                     
        for modepair in range(2):
            logger.debug(modepair)
            this_time, (alpha_1, alpha_2), (rratio_1,rratio_2) = get_dirs_transformed(modal_sorted, chan_dofs, node, modepair)
            x1 = wind_ds['mean']
            
            if 'Wr' in target: 
                x1[x1<90]+=360
            
            x2 = x1
            
            c1=np.tile(colors[modepair], [len(alpha_1),1])
            c1[:,-1]=1-np.array(rratio_1)  
            
            c2=np.tile(colors[modepair+5], [len(alpha_2),1])
            c2[:,-1]=1-np.array(rratio_2)
            
#             plt.plot(alpha_1, ls='none',marker=',')
#             plt.plot(alpha_2, ls='none',marker=',')
#             plt.show()
            
            import matplotlib.colors
            
            color_s=matplotlib.colors.to_rgb(colors[modepair])
            cdict = {
                     'red':  ((0.0,color_s[0], color_s[0]),
                              (1.0,color_s[0], color_s[0])),
                     
                     'green':((0.0,color_s[1], color_s[1]),
                              (1.0,color_s[1], color_s[1])),
                     
                     'blue' :((0.0,color_s[2], color_s[2]),
                              (1.0,color_s[2], color_s[2])),  
                                
                     'alpha':((0.0, 0.0, 0.0),
                              (0.5, 0.3, 0.3),
                              (1.0, 1.0, 1.0))}
            cmap1 = matplotlib.colors.LinearSegmentedColormap('CustomCmap1', cdict)

            color_s=matplotlib.colors.to_rgb(colors[modepair+5])
            logger.debug(color_s)
            cdict = {
                     'red':  ((0.0,color_s[0], color_s[0]),
                              (1.0,color_s[0], color_s[0])),
                     
                     'green':((0.0,color_s[1], color_s[1]),
                              (1.0,color_s[1], color_s[1])),
                     
                     'blue' :((0.0,color_s[2], color_s[2]),
                              (1.0,color_s[2], color_s[2])),  
                                
                     'alpha':((0.0, 0.0, 0.0),
                              (0.5, 0.3, 0.3),
                              (1.0, 1.0, 1.0))}
            cmap2 = matplotlib.colors.LinearSegmentedColormap('CustomCmap2', cdict)
            if ax1 is None:
                plt.figure(tight_layout=1)
                if 0:
                    plt.hexbin(x=x1, y=alpha_1, gridsize=(360,180), bins='log', cmap=cmap1, label='$\\alpha_{{{0}}}$'.format(modepair*2+1))
                    plt.hexbin(x=x2, y=alpha_2, gridsize=(360,180), bins='log', cmap=cmap2, label='$\\alpha_{{{0}}}$'.format(modepair*2+2))
                else:
                    plt.scatter(x1,alpha_1,c=c1, marker='.', s=4, edgecolors='none',label='$\\alpha_{{{0}}}$'.format(modepair*2+1))
                    plt.scatter(x2,alpha_2,c=c2, marker='.', s=4, edgecolors='none',label='$\\alpha_{{{0}}}$'.format(modepair*2+2))
                ax1=plt.gca()
            else:
                if 0:
                    ax1.hexbin(x=x1, y=alpha_1, gridsize=(360,180), bins='log', cmap=cmap1,  label='$\\alpha_{{{0}}}$'.format(modepair*2+1))
                    ax1.hexbin(x=x2, y=alpha_2, gridsize=(360,180), bins='log', cmap=cmap2,  label='$\\alpha_{{{0}}}$'.format(modepair*2+2))
                else:  
                    ax1.scatter(x1,alpha_1,c=c1, marker='.', s=4, edgecolors='none',label='$\\alpha_{{{0}}}$'.format(modepair*2+1))
                    ax1.scatter(x2,alpha_2,c=c2, marker='.', s=4, edgecolors='none',label='$\\alpha_{{{0}}}$'.format(modepair*2+2))
            if 0:
                axes[modepair].hexbin(x=x1, y=alpha_1, gridsize=(360,180), bins='log', cmap=cmap1,  label='$\\alpha_{{{0}}}$'.format(modepair*2+1))
                axes[modepair].hexbin(x=x2, y=alpha_2, gridsize=(360,180), bins='log', cmap=cmap2,  label='$\\alpha_{{{0}}}$'.format(modepair*2+2))
            else:
                logger.debug(x1, alpha_1)
                logger.debug(x2, alpha_2)
                axes[modepair].scatter(x1,alpha_1,c=c1, marker='.', s=4, edgecolors='none',label='$\\alpha_{{{0}}}$'.format(modepair*2+1))
                axes[modepair].scatter(x2,alpha_2,c=c2, marker='.', s=4, edgecolors='none',label='$\\alpha_{{{0}}}$'.format(modepair*2+2))

            #axes[modepair].set_ylabel('Hauptrichtung der Eigenform')
            if 'Wg' in target:
                #axes[modepair].set_xlabel('Windgeschwindigkeit [\si{\metre\per\second}]')
                axes[modepair].set_xlim((0,15))
            elif 'Wr' in target:
                pass
        
        ax1.set_ylabel('Main Direction of Modeshape')            
        if 'Wg' in target:
            ax1.set_xlabel('Windgeschwindigkeit [\si{\metre\per\second}]')
            ax1.set_xlim((0,15))
        elif 'Wr' in target:
            ax1.set_xlabel('Windrichtung')
        plt.close(ax1.figure)
        return axes

    def dir_wind_histo(modal, chan_dofs, wind_ds, node='1', target='Wr_top'):

        fig1,axes=plt.subplots(2,2,sharey=True, gridspec_kw={'width_ratios':[.8,.2]},tight_layout=0)
        #ax1=fig1.gca()
        fig2=plt.figure()
        ax2=fig2.gca()
        for modepair,ax in enumerate(axes[:,1]):
            plot_histograms(modal, chan_dofs,node,modepair,ax,ax2)
            plt.close(fig2)
        
        dirs_wind(wind_ds, modal, chan_dofs, node, target ,axes = list(axes[:,0]))
        
        if 'Wg' in target:
            ax3.set_xlabel('Windgeschwindigkeit [\si{\metre\per\second}]')
            ax3.set_xlim((0,15))
        elif 'Wr' in target:
            axes[1,0].set_xlabel('Wind Direction')
         
        return fig1, axes   

            
    if 1:# sort modes into pairs while skipping a lot of results
        modal_sorted = split_modepairs(modal)
        modal_sorted.to_netcdf(os.path.join(path,'result_db/modal_{}_dirs_sorted.nc'.format(quantity)), engine='h5netcdf')
        
    modal_sorted = xr.open_dataset(os.path.join(path,'result_db/modal_{}_dirs_sorted.nc'.format(quantity)), engine='h5netcdf')   
    logger.debug(modal_sorted)

    
    nodes = ['1','4','5','6','3','2']
    node= nodes[1]
         
    for target_wind in [#'Wg', 
                        #'Wr', 
                        #'Wg_top',
                        'Wr_top'
                        ]:
        wind_ds = get_stats('wind', duration)

        return dir_wind_histo(modal_sorted, chan_dofs, wind_ds,  node, target_wind)


def subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True,
             subplot_kw=None, gridspec_kw=None, **fig_kw):

    from matplotlib.gridspec import GridSpec
    figure = plt.figure
    
    if subplot_kw is None:
        subplot_kw = {}
    if gridspec_kw is None:
        gridspec_kw = {}
        
    fig = figure(**fig_kw)
    gs = GridSpec(nrows, ncols, **gridspec_kw)

    # Create empty object array to hold all axes.  It's easiest to make it 1-d
    # so we can just append subplots upon creation, and then
    nplots = nrows*ncols
    axarr = np.empty(nplots, dtype=object)

    # Create first subplot separately, so we can share it if requested
    ax0 = fig.add_subplot(gs[0, 0], **subplot_kw)
    axarr[0] = ax0

    r, c = np.mgrid[:nrows, :ncols]
    r = r.flatten() * ncols
    c = c.flatten()
    lookup = {
            "none": np.arange(nplots),
            "all": np.zeros(nplots, dtype=int),
            "row": r,
            "col": c,
            }
    sxs = lookup[sharex]
    sys = lookup[sharey]

    # Note off-by-one counting because add_subplot uses the MATLAB 1-based
    # convention.
    for i in range(1, nplots):
        if sxs[i] == i:
            subplot_kw['sharex'] = None
        else:

            subplot_kw['sharex'] = axarr[sxs[i]]
        if sys[i] == i:
            subplot_kw['sharey'] = None
        else:
            if i // ncols+1 == i % ncols:
                subplot_kw['sharey'] = None
            else:
                subplot_kw['sharey'] = axarr[sys[i]]
        axarr[i] = fig.add_subplot(gs[i // ncols, i % ncols], **subplot_kw)

    # returned axis array will be always 2-d, even if nrows=ncols=1
    axarr = axarr.reshape(nrows, ncols)

    # turn off redundant tick labeling
    if sharex in ["col", "all"] and nrows > 1:
        # turn off all but the bottom row
        for ax in axarr[:-1, :].flat:
            for label in ax.get_xticklabels():
                label.set_visible(False)
            ax.xaxis.offsetText.set_visible(False)

    if sharey in ["row", "all"] and ncols > 1:
        # turn off all but the first column
        for ax in axarr[:, 1:].flat:
            for label in ax.get_yticklabels():
                label.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)

    if squeeze:
        # Reshape the array to have the final desired dimension (nrow,ncol),
        # though discarding unneeded dimensions that equal 1.  If we only have
        # one subplot, just return it instead of a 1-element array.
        if nplots == 1:
            ret = fig, axarr[0, 0]
        else:
            ret = fig, axarr.squeeze()
    else:
        # returned axis array will be always 2-d, even if nrows=ncols=1
        ret = fig, axarr.reshape(nrows, ncols)

    return ret
    
def strain_vs_accel(duration: pd.Timedelta, mode: int=0):
    
    
    modal_results=['frequencies',
           'damping', 
           'modal_contributions',
           ]
    
    kurt_range = (-1,4)
    
    f_range = [(0.34,0.38),(0.6,0.65),(1.2,1.4),(2.0,2.15),(3.2,3.55)][mode]
    mc_range = (0,1)
    damp_range = (0,10) 
    
    #threshold for hierarchical clustering
    threshold=0.3
    
    both_results = {}
    
    for quantity in ['accel', 'strain_rosettes']:
        if quantity == 'accel':
            rms_range = (1e-4,np.inf)
        elif quantity == 'strain_rosettes':
            rms_range = (3.5e-7,np.inf)
        
        results = get_modal_results(quantity, duration)
        
        results = results.stack(flat_modes=['time','modes'])
        
        if quantity == 'accel':
            rms_arr = results['rms'].sel(channels=['Accel_01','Accel_02']).mean(dim='channels').rename('rms_m')
            kurt_arr = results['kurtosis'].sel(channels=['Accel_01','Accel_02']).mean(dim='channels').rename('kurtosis_m')
        elif quantity == 'strain_rosettes':
            rms_arr = results['rms'].sel(channels=['A_z','B_z', 'C_z', 'D_z']).mean(dim='channels').rename('rms_m')   
            kurt_arr = results['kurtosis'].sel(channels=['A_z','B_z', 'C_z', 'D_z']).mean(dim='channels').rename('kurtosis_m')
            
        ds = xr.merge([results, rms_arr, kurt_arr])
        error_ind = ds['error'].any(dim='channels')
        #ds = ds.where(np.logical_not(error_ind))
        #ds = ds.where(error_ind)
        #ds = xr.merge([ds, kurt_arr]) 
        if rms_range is not None:
              
            rms_ind = np.logical_and(rms_arr>rms_range[0],rms_arr<rms_range[1])
    
            results = results.where(rms_ind)
            
        if kurt_range is not None:
              
            kurt_ind = np.logical_and(kurt_arr>kurt_range[0],kurt_arr<kurt_range[1])
    
            results = results.where(kurt_ind)

        modal_ind = results['frequencies'].isnull()
        modal_ind = np.logical_not(modal_ind)
        if f_range is not None:
            f_ind = np.logical_and(results['frequencies']>f_range[0],results['frequencies']<f_range[1])
            modal_ind = np.logical_and(f_ind, modal_ind)
        if mc_range is not None:
            #strain_results['modal_contributions']=strain_results['modal_contributions']*strain_results['frequencies']**2
            mc_ind = np.logical_and(results['modal_contributions']>mc_range[0], results['modal_contributions']<mc_range[1])
            modal_ind = np.logical_and(modal_ind, mc_ind)
        if damp_range is not None:
            damp_ind = np.logical_and(results['damping']>damp_range[0], results['damping']<damp_range[1])
            modal_ind = np.logical_and(modal_ind, damp_ind)
        
        
#         frequencies = results['frequencies'].where(modal_ind).dropna(dim='modes',how='all').data
#         frequencies[np.isnan(frequencies)]=0
#         damping = results['damping'].where(modal_ind).dropna(dim='modes',how='all').data
#         damping[np.isnan(damping)]=0
#         modal_contributions = results['modal_contributions'].where(modal_ind).dropna(dim='modes',how='all').data
#         modal_contributions[np.isnan(modal_contributions)]=0
#         modeshapes = results['modeshapes'].where(modal_ind).dropna(dim='modes',how='all').data
#         modeshapes[np.isnan(modeshapes)]=0
#         
#         #print(frequencies.shape)
#         #print(frequencies.shape, modeshapes.shape)
#         #modeshapes = np.swapaxes(modeshapes,0,1)
#         modeshapes = np.swapaxes(modeshapes,0,2)
#         #print(frequencies.shape)
#         #print(frequencies.shape, modeshapes.shape)
#         measurement_dummy = np.zeros((100,modeshapes.shape[0]))
#         prep_data_dummy = PreProcessSignals(measurement_dummy, 10)
#         modal_data_dummy = SSIDataMC(prep_data_dummy)
#         modal_data_dummy.eigenvalues = np.zeros_like(frequencies, dtype=complex)
#         modal_data_dummy.eigenvalues.imag = frequencies
#         modal_data_dummy.modal_frequencies = frequencies
#         modal_data_dummy.modal_damping = damping
#         modal_data_dummy.mode_shapes = modeshapes
#         modal_data_dummy.modal_contributions = modal_contributions
#         modal_data_dummy.max_model_order = frequencies.shape[0]
#         
#         stabil_data = StabilCluster(modal_data_dummy, prep_data_dummy)
#         stabil_data.calculate_soft_critera_matrices()
#         stabil_plot = StabilPlot(stabil_data)
#         start_stabil_gui(stabil_plot, modal_data_dummy)
        #modal_frequencies
        #modal_damping
        #max_model_order
        #mode_shapes
        #modal_contributions
        
        #from pykalman import KalmanFilter
        #print(frequencies.shape)
        #num_modes = frequencies.shape[1]
        #frequencies = np.ma.array(frequencies)
        #frequencies.mask = np.isna(frequencies)
        #kf = KalmanFilter(n_dim_state=num_modes, n_dim_obs=num_modes)
        #kf.em(frequencies)
        
        
        results = results.where(modal_ind).dropna(dim='flat_modes',how='all')
        
        time_stamps = results['frequencies'].time.data
        frequencies = results['frequencies'].data
        damping =results['damping'].data
        modal_contributions =results['modal_contributions'].data
        if quantity =='accel':
            modeshapes = results['modeshapes'].data
        elif quantity == 'strain_rosettes':
            modeshapes = results['modeshapes'].data#.sel(channels=['A_z','B_z','C_z','D_z']).data
               
        
        return_indices = assign_modes(time_stamps,frequencies, modeshapes,threshold, damping, modal_contributions)

        both_results[quantity]=[results,  return_indices]
    
    #return
    
    #plt.show()
    
    for submode in range(2):
        strain_results,  sub_mode_inds_s = both_results['strain_rosettes']
        sub_mode_inds_s = sub_mode_inds_s[submode]
        strain_results = strain_results.where(sub_mode_inds_s)
        strain_results = strain_results.unstack(dim='flat_modes')
        
        accel_results,  sub_mode_inds_a = both_results['accel']
        sub_mode_inds_a = sub_mode_inds_a[submode]
        accel_results = accel_results.where(sub_mode_inds_a)
        accel_results = accel_results.unstack(dim='flat_modes')
        
        
        strain_results, accel_results = xr.align(strain_results, accel_results, exclude=['channels','modes'])
        
        import seaborn as sns
        
        strain_results
        for fignum,modal_result in enumerate(modal_results):
            plt.figure(fignum)
            y = strain_results[modal_result].data
            x = accel_results[modal_result].data
    
            x = np.repeat(np.expand_dims(x, axis=2), repeats=y.shape[1], axis=2)            
            y = np.repeat(np.expand_dims(y, axis=2), repeats=x.shape[1], axis=2)   
                         
            y = y.flatten()
            x = x.flatten()
            
            # remove nans
            ind = np.logical_or(np.isnan(x), np.isnan(y))
            ind = np.logical_not(ind)            
            y = y[ind]
            x = x[ind] 
            
            plt.plot(x,y, ls='none', marker='.', markersize=1, color=['red','blue'][submode])
            plt.ylabel(modal_result+' strain')
            plt.xlabel(modal_result+' accel')
            #sns.kdeplot(x, y,  ax=plt.gca(), zorder=1)
    
    for fignum in range(3):
        plt.figure(fignum)
        ylim = list(plt.ylim())
        xlim = list(plt.xlim())
        
        ylim[0] = min(ylim[0],xlim[0])
        ylim[1] = max(ylim[1],xlim[1])
        plt.ylim(ylim)
        plt.xlim(ylim)
        
    plt.show()
    
def assign_modes(time_stamps, frequencies, modeshapes, threshold, damping=None, modal_contributions=None):
    #from sklearn.neighbors import kneighbors_graph
    #from sklearn.cluster import AgglomerativeClustering
    logger.debug(threshold)
    plot_ = False
    cluster_assignments = np.zeros_like(frequencies, dtype=int)
    
    #strain 1-mac
    # [0.357,        0.34,      0.357,       0.09,      0.45 ],# strain f+mac 
    #[0.35,        0.2,      0.5,       0.3,      0.2 ],#accel 1-mac
    #plot_bounds = [(0.305,0.405),(0.575,0.675),(1.155,1.495),(1.915,2.2),(2.845,3.91)][0]
    cluster=1
    con = 1
    c_method='ward'
    
    if not cluster:
        # walk over time and f
        # assign all values of each unique start time to cluster 0, 1, 2 
        # 
        unique, indices, unique_counts = np.unique(time_stamps, return_counts=True, return_index=True)
        
        
        for count, start_time in zip(unique_counts,unique):
            logger.debug(count, start_time)
    #                 if count ==1:
    #                     st=start_time
    #                     result_folder = os.path.join(path, 'modal_{}'.format(quantity),'{}'.format(st.year),'{:02d}'.format(st.month))
    #                     figure_name = os.path.join(result_folder, 
    #                         '{:02d}_{:02d}-{:02d}_{:02d}-{:02d}_stabil.png'.format(
    #                              st.year, st.month, st.day, st.hour, st.minute))
    #                     from PIL import Image
    #                     img = Image.open(figure_name)
    #                     img.show()
    #                     input('continue?')
                
            if count == 1: continue
            
            inds = time_stamps == start_time
            
            
            
            while count > 2:
                MC = modal_contributions[inds]
                # remove lowest Modal Contributions
                ind = np.argmin(MC)
                #cluster_assignments[inds][ind] = 0
                
                #set inds corresponding to ind to 0
                this_inds = inds[inds]
                this_inds[ind]=False
                inds[inds]=this_inds
                count = np.sum(inds)
    
            if count == 2:
                f = frequencies[inds]
                this_cluster_assignments = cluster_assignments[inds]
                
    
                ind = np.argmin(f)
                this_cluster_assignments[ind]=1
                ind = np.argmax(f)
                this_cluster_assignments[ind]=2
                cluster_assignments[inds] = this_cluster_assignments
    
            else:
                raise
            
            select_clusters=[1,0,0]
            
    else:
        l = frequencies.shape[0]         
        
        #knn_graph = kneighbors_graph(X=time_stamps, n_neighbors, mode, metric, p, metric_params, include_self, n_jobs)
        time_stamps = np.array(time_stamps)
        time_proximity_matrix = (time_stamps - time_stamps.reshape((l, 1)))==np.timedelta64(0)
        logger.debug(np.sum(time_proximity_matrix))
        
        mac_proximity_matrix = 1 - \
            StabilCalc.calculateMAC(modeshapes, modeshapes)
        #con = 0
        if con:
            mac_proximity_matrix[time_proximity_matrix]=1
        
        proximity_matrix = mac_proximity_matrix
        
        
        proximity_matrix[
                proximity_matrix < np.finfo(proximity_matrix.dtype).eps] = 0
        #c_method = 'ward'
        proximity_matrix_sq = scipy.spatial.distance.squareform(
            proximity_matrix, checks=False)
        linkage_matrix = scipy.cluster.hierarchy.linkage(
            proximity_matrix_sq, method=c_method)
        
        
        leaves = scipy.cluster.hierarchy.leaves_list(linkage_matrix)
        if plot_:
            fig = plt.figure(tight_layout=1)
            ax = fig.add_subplot(111)
            
            scipy.cluster.hierarchy.dendrogram(
                    linkage_matrix, 
                    #p=150,
                    #truncate_mode='lastp',
                    leaf_label_func=lambda x: '', 
                    color_threshold=threshold, 
                    leaf_font_size=16, leaf_rotation=40, 
                    #show_leaf_counts=True
                    )
            ax = plt.gca()
            ax.set_xlabel('Mode number [-]')
            ax.set_ylabel('Distance [-]')

        threshold=2
        criterion='maxclust'
        cluster_assignments = scipy.cluster.hierarchy.fcluster(
            linkage_matrix, 
            threshold, 
            criterion)  
        
        num_clusters =max(cluster_assignments)+1
        logger.debug(num_clusters)
        bars= np.array([sum(cluster_assignments==i) for i in range(num_clusters)])
        _, select_clusters = scipy.cluster.vq.kmeans2(
                    np.array(bars, dtype=np.float64), 
                    np.array([max(bars), 1e-12]))
        if plot_:
            plt.figure()
            x=np.array(np.arange(num_clusters))
    
            plt.bar(x[np.array(select_clusters, dtype=bool)], bars[np.array(select_clusters, dtype=bool)], color='blue')
            plt.bar(x[np.logical_not(select_clusters)], bars[np.logical_not(select_clusters)], color='red')
    
    
    total_modes_clustered=0
    mshs=[]
    times=[]
    fs=[]
    return_indices = []
    mean_f = []
    for i,inout in enumerate(select_clusters):
        if inout: continue
        this_cluster = cluster_assignments == i
        cluster_size = sum(this_cluster)
        time_stamps_ = time_stamps[this_cluster]
        frequencies_ = frequencies[this_cluster]
        damping_ = damping[this_cluster]
        modal_contributions_ = modal_contributions[this_cluster]
        modeshapes_ = modeshapes[:,this_cluster]
        
        # remove multiple values from the same dataset (=start_time)
        unique, indices, unique_counts = np.unique(time_stamps_, return_counts=True, return_index=True)            
        inds= indices[unique_counts==1]
        
        full_inds = np.zeros_like(frequencies, dtype=bool)
        full_inds_2 = full_inds[this_cluster]
        full_inds_2[inds] = True
        full_inds[this_cluster]=full_inds_2
        
        return_indices.append(full_inds)
        
        
        logger.info('{} unique time stamps out of {} timestamps total'.format(np.sum(unique_counts==1), len(time_stamps_)))
        time_stamps_ = time_stamps[full_inds]
        frequencies_ = frequencies[full_inds]
        damping_ = damping[full_inds]
        modal_contributions_ = modal_contributions[full_inds]
        modeshapes_ = modeshapes[:,full_inds]
        mshs.append(modeshapes_)
        times.append(time_stamps_)
        fs.append(frequencies_)
        this_mac = StabilCalc.calculateMAC(modeshapes_, modeshapes_)
        
        logger.debug('Frequency: {:1.3f}+-{:1.3f} Hz'.format(frequencies_.mean(), 2*frequencies_.std()))
        logger.debug('MAC: {:1.3f}+-{:1.3f}'.format(this_mac.mean(), this_mac.std()))
        mean_f.append(frequencies_.mean())
        
        total_modes_clustered += len(time_stamps_)            
        if 0:#plot_:
            fig=plt.figure(figsize=(6,3))
            fig.suptitle('Clustersize: {}; Frequency: {:1.3f}+-{:1.3f} Hz'.format(len(frequencies_), frequencies_.mean(), 2*frequencies_.std()))
            ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
            ax3 = plt.subplot2grid((2, 2), (0, 1))
            ax4 = plt.subplot2grid((2, 2), (1, 1), sharex=ax3)            
            
            ax1.matshow(this_mac, vmin=0, vmax=1)
            ax3.plot(time_stamps_, frequencies_, ls='none', marker='.', markersize=1)
            #ax3.set_ylim(plot_bounds)            
        #             ax4.plot(time_stamps_, damping_, ls='none', marker='.', markersize=1)
        #             ax4.set_ylim((0,5))
            ax4.plot(time_stamps_, modal_contributions_, ls='none', marker='.', markersize=1)
            ax4.set_ylim((0,0.5))
    
    #plt.figure()
    unique, counts = np.unique(time_stamps, return_counts=True)

    logger.info('Clustered {} out of {} ({} %) modes'.format(total_modes_clustered, len(unique)*2, total_modes_clustered/len(unique)/2))
    
    if plot_:
        plt.show()
    if mean_f[0]>mean_f[1]:
        return_indices.reverse()
    return return_indices
    
        
    if plot_:
        unique, counts = np.unique(np.hstack(times), return_counts=True)
    
        times_=[]
        dfs=[]
        for ts, count in zip(unique, counts):
            ind1=times[0]==ts
            ind2=times[1]==ts
            if ind1.any() and ind2.any():
                f_1=fs[0][ind1]
                f_2=fs[1][ind2]
        
                times_.append(ts)
                dfs.append(f_1-f_2)
        plt.figure()
        plt.plot(times_,dfs, ls='none', marker='.')
        
        
        mac=StabilCalc.calculateMAC(mshs[0], mshs[1])
        plt.matshow(mac, vmin=0, vmax=1)
        plt.show()
        
def test():
    quantity = 'accel'
    origin = config.origins[quantity]
    ds =  get_file_info(origin)
    day = np.datetime64('2024-03-19 03:00')
    filename = ds.sel(time=day)['file_name'].data.item()
    yn = input('read and plot {} (y/n)'.format(filename))
    if yn == 'y':
        filepath = os.path.join(config.file_root_path, config.subpaths[origin], filename)
        file_contents = read_file(filepath)
        if file_contents is None:
            logger.warning('File unreadable: {}'.format(filepath))
            return
        file_time, file_size, headers, units, start_time, sample_rate,measurement = file_contents
        
        plot_file(file_time, headers, units, start_time, sample_rate, measurement)
        
def main():
    ###################################################################
    # Input the following:                                            #
    ###################################################################
    
    # set the root directory where the database files are stored
    # that directory should contain file_info_<quanity>.nc at its root
    # and subfolders with stats_<>.nc/modal_<>.nc for each block length in minutes
    config.db_root_path = '/home/towermonitoring/analysis/result_db/'
    
    # define the length of signal blocks to analyse
    minutes = 120 # in minutes, must be one of 10, 30, 60, 120
    # define the quantity to use for postprocessing (not all plots)
    quantity = 'accel' # must be one of 'accel', 'wind', 'temp', 'strain_rosettes'
    # define if plots should be saved or shown
    save_figures=False
    # define the path were figures should be saved
    figpath = '/vegas/users/staff/womo1998/Projects/2023_EVACES/current_figures/'
    # should LaTeX be used to render figure labels and numbers?
    use_tex = False
    
    # select plots to draw (toogle with 
    its = False# inspect_time_shifts
    pfi = True# plot_file_info
    ps = False# plot_stats
    pld = False# plot_daily
    day = np.datetime64('2024-03-19 00:00')
    pmr = False# postprocess_modal_results (scatter plot)
    ft = False# frequencies over time
    fT = False#frequencies over temp
    dw = False#damping over wind
    modrms=False#modal parameters over rms
    icov = False#icing overview
    ic = False#icing events
    fr = False# frequencies over rms
    cd = False# cov_freq over data_quality !!possibly broken code!!
    ck = False# cov_freq over kurtosis !!possibly broken code!!
    dd = False# main_dir over wind_dir !!possibly broken code!!
    
    ####################################################################
    
    logger.setLevel(logging.INFO)
    
    assert quantity in ['accel', 'wind', 'temp', 'strain_rosettes']
    
    assert minutes in [10,30,60,120]
    duration = pd.Timedelta(minutes=minutes)
    minutes = int(duration.total_seconds()/60)
    
    locale.setlocale(locale.LC_ALL,'de_DE.utf8')
    print_context_dict ={'text.usetex':use_tex,
                        'font.size':9,                                                                                 
                         'legend.fontsize':9, 
                         'legend.labelspacing':0.1,                                                       
                         'axes.linewidth':0.5,                                                                           
                         'xtick.major.width':0.2,                                                                        
                         'ytick.major.width':0.2,                                                                        
                         'text.latex.preamble':r"\usepackage{siunitx}\usepackage[utf8]{inputenc}\usepackage{nicefrac}",                                                                        
                         'xtick.major.width':0.5,                                                                        
                         'ytick.major.width':0.5, 
                         'figure.figsize':(5.906, 5.906 / 1.618),  # print #150 mm \columnwidth
                         # 'figure.figsize':(5.906/2,5.906/2/1.618),#half print #150 mm \columnwidth
                         # 'figure.figsize':(5.53,2.96),# beamer
                         # 'figure.figsize':(5.53/2,2.96),# half beamer
                         'figure.dpi':300}

    if its:
        inspect_time_shifts()
        return

    origin = config.origins[quantity]
    logger.info('{}, {}'.format(quantity, origin))
    
    if False:
        plot_waterfall(quantity, duration, pd.Timestamp('2023-01-21T01:00').to_datetime64())
        plt.show()
        
    if pfi:
        plot_file_info(origin, check_errors=1, filter_errors=0)
        plt.show()

    if ps:
        plot_stats(quantity, duration, check_errors=0, filter_errors=0)
        plt.show()
    
    if ps:
        plot_stats(quantity, duration, check_errors=0, filter_errors=0, modal=True)
        plt.show()
    
    if pld:
        plot_daily(quantity, duration, day)
        plt.show()
    
    if pmr and quantity in ['accel', 'strain_rosettes']:
        
        for wind_range in range(1):
            for temp_range in range(1):
                for mode in range(6):
                    wind_str =['all',     'weak','moderate','strong','custom'][wind_range]
                    temp_str =[ 'all',    'frost'    , '0-10','10-20','hot'][temp_range]
                    mode_str = ['all','first','second','third', 'fourth', 'fifth'][mode]
                    
                    if os.path.exists(figpath + 'q_{}-m_{}-t_{}-w_{}.png'.format(quantity, mode_str, temp_str, wind_str)):
                        logger.debug('q_{}-m_{}-t_{}-w_{}'.format(quantity, mode_str, temp_str, wind_str))
                        #continue
                    if quantity == 'accel':
                        color='#006B94'
                    elif quantity =='strain_rosettes':
                        color='maroon'
                    else:
                        color='black'
                        
                    with matplotlib.rc_context(rc=print_context_dict):
                        
                        postprocess_modal_results(quantity, duration,
                                                  filter_errors=True, 
                                                  wind_range=wind_range, 
                                                  temp_range=temp_range, 
                                                  mode_pair=mode,
                                                  damp_range=(0,5),
                                                  color=color, 
                                                  hide_ticks=True, 
                                                  scatter=False,
                                                  )
                    if save_figures:
                        plt.gcf().savefig(figpath + 'q_{}-d_{}-m_{}-t_{}-w_{}.png'.format(quantity,minutes, mode_str, temp_str, wind_str))
                    else:
                        plt.show()

    # frequencies over time
    if ft:
        # print_context_dict['figure.figsize']=(5.53*0.62,2.96)
        with matplotlib.rc_context(rc=print_context_dict):
            mode=0
            q_1=['frequencies']
            q_2=['time']
            fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
            plt.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.14, hspace=0.12)
            for i,_quantity in enumerate(['accel','strain_rosettes']):
                if _quantity == 'accel':
                    color='#006B94'
                elif _quantity =='strain_rosettes':
                    color='maroon'
                else:
                    color='black'
                postprocess_modal_results(_quantity, duration, filter_errors=True,
                                          wind_range=0, temp_range=0, mode_pair=mode, q_1=q_1, q_2=q_2,
                                          fig=fig, axes=axes[i], color=color, scatter=False)
                mode_str = ['all','first','second','third', 'fourth', 'fifth'][mode]
                axes[i].set_ylabel('')
                plt.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.14, hspace=0.12)
                axes[i].yaxis.set_ticks_position('left')
            
            
            months = MonthLocator(range(1,13,6), bymonthday=1, interval=1)
            monthsFmt = DateFormatter("%b '%y")
            axes[1].xaxis.set_major_locator(months)
            axes[1].xaxis.set_major_formatter(monthsFmt)
            axes[1].set_xlabel('')
            fig.text(0.01, 0.55, 'Frequency [\si{\hertz}]', va='center', rotation='vertical')
            
            if save_figures:
                plt.gcf().savefig(figpath + f'q_all-d_{minutes}-freq_vs_time.png', dpi=300)
                plt.close('all')
            else:
                plt.show()
    
    #frequencies over temp
    if fT:
        # print_context_dict['figure.figsize']=(5.53*0.62,2.96)
        with matplotlib.rc_context(rc=print_context_dict):
            fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=False)
            if quantity == 'accel':
                color='#006B94'
            elif quantity =='strain_rosettes':
                color='maroon'
            else:
                color='black'
                
            for i,mode in enumerate(reversed([1,2,3,4,5])):
                # call this once before, to ensure even hexbins (axes sizes must be known before plotting)
                plt.subplots_adjust(left=0.1, right=0.98, top=0.975, bottom=0.115)
                q_1=['frequencies']
                q_2=['temp']
                postprocess_modal_results(quantity, duration, filter_errors=False, 
                                          wind_range=0, temp_range=(-10,35), mode_pair=mode, q_1=q_1, q_2=q_2, 
                                          fig=fig,axes=axes[i], color=color, scatter=False, )

                mode_str = ['all','first','second','third', 'fourth', 'fifth'][mode]
                axes[i].set_ylabel('')
                axes[i].yaxis.set_ticks_position('left')
                    
            loc = matplotlib.ticker.MultipleLocator(base=0.01) # this locator puts ticks at regular intervals
            for i in range(5):
                axes[i].yaxis.set_minor_locator(matplotlib.ticker.LinearLocator(numticks=3))
                axes[i].yaxis.set_major_locator(matplotlib.ticker.LinearLocator(numticks=3))
                axes[i].xaxis.grid(True, lw=0.1, zorder=-1)
                axes[i].yaxis.grid(True, lw=0.1, zorder=-1)
            
            plt.subplots_adjust(left=0.1, right=0.98, top=0.975, bottom=0.115)
            fig.text(0.01, 0.55, 'Frequency [\si{\hertz}]', va='center', rotation='vertical')
            axes[-1].set_xlabel('Temperature [\si{\celsius}]')

            
            
            if save_figures:
                plt.gcf().savefig(figpath + f'q_{quantity}-d_{minutes}-freq_vs_temp.png', dpi=300)
                #plt.gcf().savefig('/ismhome/staff/womo1998/Projects/2018_ISMA/paper/figures/three_modes_temp.pdf', dpi=300)
                plt.close('all')
            else:
                plt.show()
    
    #damping over wind
    if dw:
        # print_context_dict['figure.figsize']=(5.53*0.62,2.96)
        with matplotlib.rc_context(rc=print_context_dict):
            modes = [1,2,5]
            fig, axes = plt.subplots(nrows=len(modes), ncols=1, sharex=True, sharey=True)
            fig.subplots_adjust(left=0.075, right=0.98, top=0.965, bottom=0.115, hspace=0.165)
            
            if quantity == 'accel':
                color='#006B94'
            elif quantity =='strain_rosettes':
                color='maroon'
            else:
                color='black'
            
            # for color,quantity in list(zip(['#006B9455','maroon'],['accel','strain_rosettes']))[:1]:
            for i,mode in enumerate(reversed(modes)):
                q_1=['damping']
                q_2=['wind']
                #continue
                postprocess_modal_results(quantity, duration, filter_errors=True, damp_range=(0,2.5),
                                          wind_range=4, temp_range=0, mode_pair=mode, q_1=q_1, q_2=q_2, 
                                          fig=fig,axes=axes[i], color=color,scatter=False, hexbin_extent=[0,20,0,2.5])
                #plt.show()
                mode_str = ['all','first','second','third', 'fourth', 'fifth'][mode]
                axes[i].set_ylabel('')
                #plt.gcf().suptitle('{}: {} mode(s), x: {}, y: {}'.format(quantity, mode_str, q_2[0], q_1[0]))
                #plt.gcf().set_size_inches(6.3,3.9/2)
                axes[i].yaxis.set_ticks_position('left')
                fig.subplots_adjust(left=0.075, right=0.98, top=0.965, bottom=0.115, hspace=0.165)
                    
            #loc = matplotlib.ticker.MultipleLocator(base=0.01) # this locator puts ticks at regular intervals
            for i in range(len(modes)):
                axes[i].yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.5))
                axes[i].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=2.5))
            
            fig.text(0.01, 0.55, 'Damping [\si{\percent}]', va='center', rotation='vertical')
            axes[-1].set_xlabel('Wind Speed [\si{\metre\per\second}]')
            axes[-1].set_xlim((-0.0001,20.0001))
            # axes[0].set_ylim((0,2.5))
                
            
            if save_figures:
                plt.gcf().savefig(figpath + f'q_{quantity}-d_{minutes}-damp_vs_wind.png', dpi=300)
                #plt.gcf().savefig('/ismhome/staff/womo1998/Projects/2018_ISMA/paper/figures/three_modes_wind.pdf', dpi=300)
                plt.close('all')
            else:
                plt.show()
    
    #modal over rms
    if modrms:
        # print_context_dict['figure.figsize']=(5.53*0.62,2.96)
        with matplotlib.rc_context(rc=print_context_dict):
            for mode_pair in range(1,6):
                fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(5.906/2, 5.906 / 1.618 ))
                fig.subplots_adjust(left=0.120, right=0.965, top=0.965, bottom=0.115, hspace=0.165)
                
                if quantity == 'accel':
                    color='#006B94'
                elif quantity =='strain_rosettes':
                    color='maroon'
                else:
                    color='black'
                
                q_2=['rms_m']
                for i, q_1 in enumerate([['damping'],['frequencies']]):
                    #continue
                    postprocess_modal_results(quantity, duration, filter_errors=True, rms_range=(0,16),
                                              damp_range=(0,5), bin_factor=3,
                                              wind_range=0, temp_range=0, mode_pair=mode_pair, q_1=q_1, q_2=q_2, 
                                              fig=fig,axes=axes[i], color=color,scatter=False)
                    #plt.show()
                    mode_str = ['all','first','second','third', 'fourth', 'fifth'][mode_pair]
                    
                    fig.subplots_adjust(left=0.120, right=0.965, top=0.965, bottom=0.115, hspace=0.165)
                
                axes[0].set_ylabel('Damping [\si{\percent}]', ha='center',rotation='vertical', labelpad=-10)
                axes[1].set_ylabel('Frequency [\si{\hertz}]', ha='center',rotation='vertical', labelpad=-10)
                
                for i in range(2):
                    axes[i].yaxis.set_minor_locator(matplotlib.ticker.LinearLocator(numticks=3))
                    axes[i].yaxis.set_major_locator(matplotlib.ticker.LinearLocator(numticks=2))
                    axes[i].xaxis.grid(True, lw=0.1, zorder=-1)
                    axes[i].yaxis.grid(True, 'minor', lw=0.1, zorder=-1)
                axes[-1].set_xlabel('$\\text{RMS} [\si{\milli\metre\per\square\second}]$')

                    
                fig.align_ylabels()
                if save_figures:
                    plt.gcf().savefig(figpath + f'q_{quantity}-d_{minutes}-d_{mode_pair}-modal_vs_rms.png', dpi=300)
                    #plt.gcf().savefig('/ismhome/staff/womo1998/Projects/2018_ISMA/paper/figures/three_modes_wind.pdf', dpi=300)
                    plt.close('all')
                else:
                    plt.show()
    # icing
    if icov:
        with matplotlib.rc_context(rc=print_context_dict):
            
            
            fig = plt.figure()
            spec = fig.add_gridspec(7,2, height_ratios=[1,1,1.14,0.86,1,1,0.28])
            axes = []
            axes.append(fig.add_subplot(spec[0:2, 0]))
            axes.append(fig.add_subplot(spec[2:4, 0],sharex=axes[-1]))
            axes.append(fig.add_subplot(spec[4:6, 0],sharex=axes[-1]))
            axes.append(fig.add_subplot(spec[0:3, 1],sharey=axes[-2]))
            axes.append(fig.add_subplot(spec[3:7, 1],sharex=axes[-1],sharey=axes[-2]))
            
            # fig, axes = plt.subplots(nrows=3, ncols=2, sharex='col', sharey='row')
            
            if quantity == 'accel':
                color='#006B94'
            elif quantity =='strain_rosettes':
                color='maroon'
            else:
                color='black'
            #for color,quantity in zip(['#006b9455','maroon'],['accel','strain_rosettes']):
            color=matplotlib.colors.to_rgba(color, alpha=0.5)
            
            plt.subplots_adjust(left=0.095, right=0.97, top=0.95, bottom=0.11, hspace=0.12, wspace=0.06)
            postprocess_modal_results(quantity, duration, filter_errors=False, 
                                      wind_range=0, temp_range=0, mode_pair=0, q_1=['temp'], q_2=['time'], 
                                      fig=fig, axes=axes[0], color=color, scatter=False)
            axes[0].axhline(0,color='black', lw=0.1)
            axes[0].xaxis.set_visible(False)
            
            #linear regressions of frequencies vs. temp
            predict_factors = [(1), #dummy
                               (-2.55343527e-04,  3.57122155e-01), #mode_pair 1
                               (-2.05728155e-04,  6.24917728e-01),#mode_pair 2
                               (3.09462067e-05, 1.30772396e+00),
                               (-1.74891107e-03,  2.07371664e+00),
                               (-2.88029598e-03,  3.38148568e+00),]
            
            for j,q_2 in enumerate([['time'],['temp']]):
                for i,mode in enumerate(reversed([1,2])):
                    q_1=['frequencies']
                    ax = axes[j * 2 + i + 1]
                    plt.subplots_adjust(left=0.095, right=0.97, top=0.95, bottom=0.11, hspace=0.12, wspace=0.06)
                    postprocess_modal_results(quantity, duration, filter_errors=False, 
                                              wind_range=0, temp_range=0, mode_pair=mode, q_1=q_1, q_2=q_2, 
                                              fig=fig,axes=ax, color=color, scatter=False)
                
                if 'time' in q_2 and False:
                    for i,mode in enumerate(reversed([1,2])):
                        q_1=['frequencies']
                        ax = axes[j * 2 + i + 1]
                        '''
                        when q_2=='time' and q1=='frequencies'
                        we need to get q_1='temp' and predict 'frequencies' from a linear regression
                        then overlay it on the actual plot
                        run the loop once again with a predict switch
                        '''
                        plt.subplots_adjust(left=0.095, right=0.97, top=0.95, bottom=0.11, hspace=0.12, wspace=0.06)
                        postprocess_modal_results(quantity, duration, filter_errors=False, 
                                                  wind_range=0, temp_range=(-40,5), mode_pair=mode, q_1=['temp'], q_2=q_2, 
                                                  fig=fig,axes=ax, color=matplotlib.colors.to_rgba('red', alpha=0.5), scatter=True, predict=predict_factors[mode])

            months = MonthLocator(range(1, 13, 6), bymonthday=1, interval=1)
            monthsFmt = DateFormatter("%b '%y")
            axes[0].xaxis.set_visible(False)
            axes[1].xaxis.set_visible(False)
            axes[2].set_xlabel('Time')
            axes[2].xaxis.set_visible(True)
            axes[2].xaxis.set_major_locator(months)
            axes[2].xaxis.set_major_formatter(monthsFmt)
            rotation=30 
            ha='right'
            for label in axes[2].get_xticklabels():
                #continue
                label.set_ha(ha)
                label.set_rotation(rotation)
                
            loc = matplotlib.ticker.MultipleLocator(base=0.01) # this locator puts ticks at regular intervals
    
            for i in range(1,3):
                axes[i].yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.01))
                axes[i].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.02))
    #         
            fig.text(0.01, 0.4, 'Frequency [\si{\hertz}]', va='center', rotation='vertical')
            
            axes[3].set_ylabel('')
            axes[3].yaxis.set_visible(False)
            axes[4].set_xlabel('Temperature [\si{\celsius}]')
            axes[4].set_xlim((-18,35))
            axes[4].set_ylabel('')
            axes[4].yaxis.set_visible(False)
            axes[0].set_ylabel('Temperature [\si{\celsius}]', va='center', ha='center', rotation='vertical')
            axes[0].set_ylim((-18,35))
            axes[1].set_ylabel('')
            axes[2].set_ylabel('')
            
            
            plt.subplots_adjust(left=0.095, right=0.97, top=0.95, bottom=0.11, hspace=0.12, wspace=0.06)
            fig.align_xlabels()
            
            figwidth, figheight = fig.get_size_inches()
            figwidth *= fig.get_dpi()
            figheight *= fig.get_dpi()
            x0,y0,width,height = axes[1].bbox.bounds
            ax1tr = ((x0 + width)/figwidth, (y0+height)/figheight)
            ax1br = ((x0 + width)/figwidth, (y0)/figheight)
            
            x0,y0,width,height = axes[2].bbox.bounds
            ax2tr = ((x0+width)/figwidth, (y0+height)/figheight)
            ax2br = ((x0+width)/figwidth, (y0)/figheight)
            
            x0,y0,width,height = axes[3].bbox.bounds
            ax3tl = (x0/figwidth, (y0+height)/figheight)
            ax3bl = (x0/figwidth, (y0)/figheight)
            
            x0,y0,width,height = axes[4].bbox.bounds
            ax4tl = (x0/figwidth, (y0+height)/figheight)
            ax4bl = (x0/figwidth, (y0)/figheight)
            logger.debug(axes[0].xaxis)
            # l = matplotlib.lines.Line2D(*zip((ax1tr,ax3tl)), lw=5., alpha=0.3)
            l = matplotlib.lines.Line2D([ax1tr[0],ax3tl[0]],[ax1tr[1],ax3tl[1]], lw=0.5, color='k')
            fig.add_artist(l)
            l = matplotlib.lines.Line2D([ax1br[0],ax3bl[0]],[ax1br[1],ax3bl[1]], lw=0.5, color='k')
            fig.add_artist(l)
            l = matplotlib.lines.Line2D([ax2br[0],ax4bl[0]],[ax2br[1],ax4bl[1]], lw=0.5, color='k')
            fig.add_artist(l)
            l = matplotlib.lines.Line2D([ax2tr[0],ax4tl[0]],[ax2tr[1],ax4tl[1]], lw=0.5, color='k')
            fig.add_artist(l)
            
            if save_figures:
                plt.gcf().savefig(figpath + f'q_{quantity}-d_{minutes}-icing.png', dpi=300)
                plt.close('all')
            else:
                plt.show()
            
    # icing
    if ic:
        with matplotlib.rc_context(rc=print_context_dict):
            
            for n in range(6):
                icing_events = [
                    ('2016-11-25' ,   '2016-12-10'),
                    ('2016-12-27' ,   '2017-01-25'),
                    ('2017-11-25' ,   '2017-12-17'),
                    ('2020-12-29' ,   '2021-02-01'),
                    ('2022-01-03',    '2022-02-16'),
                    ('2023-01-13' ,   '2023-02-13')]
                icing_event = [pd.Timestamp(event) for event in icing_events[n]]
                
                fig, axes = plt.subplots(3,1, sharex=True)
                
                # fig, axes = plt.subplots(nrows=3, ncols=2, sharex='col', sharey='row')
                
                if quantity == 'accel':
                    color='#006B94'
                elif quantity =='strain_rosettes':
                    color='maroon'
                else:
                    color='black'
                
                plt.subplots_adjust(left=0.1, right=0.98, top=0.97, bottom=0.135, hspace=0.12, wspace=0.06)
                postprocess_modal_results(quantity, duration, filter_errors=False, 
                                                  time_range = icing_event,
                                          wind_range=0, temp_range=0, mode_pair=0, q_1=['temp'], q_2=['time'], 
                                          fig=fig, axes=axes[0], color=color, scatter=True)
                axes[0].axhline(0,color='black', lw=0.1)
                # axes[0].xaxis.set_visible(False)
                
                #linear regressions of frequencies vs. temp
                predict_factors = [(1), #dummy
                                   (-2.55343527e-04,  3.57122155e-01), #mode_pair 1
                                   (-2.05728155e-04,  6.24917728e-01),#mode_pair 2
                                   (3.09462067e-05, 1.30772396e+00),
                                   (-1.74891107e-03,  2.07371664e+00),
                                   (-2.88029598e-03,  3.38148568e+00),]
                
                for i,mode in enumerate(reversed([1,2])):
                    q_1=['frequencies']
                    ax = axes[i + 1]
                    plt.subplots_adjust(left=0.1, right=0.98, top=0.97, bottom=0.135, hspace=0.12, wspace=0.06)
                    postprocess_modal_results(quantity, duration, filter_errors=False, 
                                              time_range = icing_event,
                                              wind_range=0, temp_range=0, mode_pair=mode, q_1=q_1,q_2=['time'], 
                                              fig=fig,axes=ax, color=color, scatter=True)
                for ax in axes:
                    ax.xaxis.grid(True, lw=0.1, zorder=-1)
                    ax.yaxis.grid(True, lw=0.1, zorder=-1)
    
                months = MonthLocator(range(1, 13, 6), bymonthday=1, interval=1)
                monthsFmt = DateFormatter("%b '%y")
                # axes[0].xaxis.set_visible(False)
                # axes[1].xaxis.set_visible(False)
                axes[2].set_xlabel('Time')
                axes[2].xaxis.set_visible(True)
                # axes[2].xaxis.set_major_locator(months)
                # axes[2].xaxis.set_major_formatter(monthsFmt)
                rotation=30 
                ha='right'
                for label in axes[2].get_xticklabels():
                    #continue
                    label.set_ha(ha)
                    label.set_rotation(rotation)
                    
                loc = matplotlib.ticker.MultipleLocator(base=0.01) # this locator puts ticks at regular intervals
        
                for i in range(1,3):
                    axes[i].yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.01))
                    axes[i].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.02))
        #         
                fig.text(0.01, 0.4, 'Frequency [\si{\hertz}]', va='center', rotation='vertical')
                
                axes[0].set_ylabel('Temperature [\si{\celsius}]', va='center', ha='center', rotation='vertical')
                axes[0].set_ylim((-18,35))
                axes[1].set_ylabel('')
                axes[2].set_ylabel('')
                
                
                plt.subplots_adjust(left=0.1, right=0.98, top=0.97, bottom=0.135, hspace=0.12, wspace=0.06)
                fig.align_xlabels()
                
                
                if save_figures:
                    plt.gcf().savefig(figpath + f'q_{quantity}-d_{minutes}-icing_event_{n}.png', dpi=300)
                    plt.close('all')
                else:
                    plt.show()
                    
    # frequencies over rms
    if fr:
        with matplotlib.rc_context(rc=print_context_dict):
            fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
            fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.12, hspace=0.04)
            mode=0
            for i,(color,_quantity) in enumerate(zip(['#006b94','maroon'],['accel','strain_rosettes'])):
                q_2=['frequencies']
                q_1=['rms_m']
                ylim = [(0,24),(0,70)][i]
                postprocess_modal_results(_quantity, duration, filter_errors=True, rms_range=ylim,
                                          wind_range=0, temp_range=0, mode_pair=mode, q_1=q_1, q_2=q_2, 
                                          fig=fig,axes=axes[i], color=color, scatter=False)
                fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.12, hspace=0.04)
            
            axes[0].set_xticks([0.35,0.62,1.31,2.06,3.36])
            axes[1].set_xticks([0.35,0.62,1.31,2.06,3.36])
            axes[0].grid(True, 'major','x', zorder=0, lw=0.1)
            axes[1].grid(True, 'major','x', zorder=0, lw=0.1)
            if use_tex:
                axes[0].set_ylabel('$\\text{RMS}_{\\text{accel}} [\si{\milli\metre\per\square\second}]$', ha='center',rotation='vertical')    
                axes[1].set_ylabel('$\\text{RMS}_{\\text{strain}}[\si{\micro\metre\per\metre}]$', ha='center',rotation='vertical')   
            axes[1].set_xlabel('Frequency [\si{\hertz}]')
            fig.align_ylabels()

            if save_figures:
                fig.savefig(figpath + f'q_all-d_{minutes}-freq_vs_rms.png', dpi=300)
                #plt.gcf().savefig('/ismhome/staff/womo1998/Projects/2018_ISMA/paper/figures/freq_vs_rms.pdf', dpi=300)
                plt.close('all')
            else:
                plt.show()    
    
    # cov_freq over data_quality
    if cd:
        # print_context_dict['figure.figsize']=(5.53*0.5,2.96)
        with matplotlib.rc_context(rc=print_context_dict):
            fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
            quantity='strain_rosettes'
            mode=0
            log_scale = True
            for i,(color,quantity) in enumerate(zip(['#006b9455','maroon'],['accel','strain_rosettes'])):

                q_1=['cov_frequencies']
                q_2=['data_quality']

                postprocess_modal_results(quantity, duration, filter_errors=False, 
                                          wind_range=0, temp_range=0, mode_pair=mode, 
                                          q_1=q_1, q_2=q_2, 
                                          fig=fig,axes=axes[i], color=color,scatter=False,
                                          hexbin_extent = [11,82,-4.5,0], 
                                          log_scale=log_scale)

                mode_str = ['all','first','second','third', 'fourth', 'fifth'][mode]
                #axes[i].set_ylabel('')

                plt.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.14, hspace=0.04)
                #axes[i].yaxis.set_ticks_position('left')
            if log_scale:
                axes[0].set_ylim((10**-4.5,1e0))
                axes[0].set_xlim((1e11,1e82))
                axes[0].set_yscale('log')
                axes[0].set_xscale('log')
            else:
                axes[0].set_ylim((-4.5,0))
                axes[0].set_xlim((11,82))
                
            
            axes[0].yaxis.set_visible(True)
            axes[0].yaxis.set_ticks_position('left')
            axes[0].set_ylabel('')
            axes[0].set_xticks([1e11,1e23,1e35,1e47,1e59,1e71])
            
            axes[0].set_xticklabels(['11','23','35','47','59','71'])
            axes[1].set_ylabel('')
            axes[1].set_xlabel('$\Delta_{\sigma_{PSD}}[\si{\decibel}]$')
            
            #fig.text(0.01, 0.55, '$\Delta_{\sigma_{PSD}}[\si{\decibel}]$', va='center', rotation='vertical')
            fig.text(0.01, 0.55, '$c_{v_f} [-]$', va='center', rotation='vertical')
            #axes[1].set_ylabel('$\Delta_{\sigma_{PSD}}[\si{\decibel}]$', ha='center',rotation='vertical', labelpad=1)   

            axes[1].yaxis.set_visible(True)
            axes[1].yaxis.set_ticks_position('left')
            #axes[1].set_ylabel('$c_{v_{f}} [-]$')
            
            
            if save_figures:
                plt.gcf().savefig(figpath + 'cov_freq_vs_data_qual_pres.png', dpi=300)
                #plt.gcf().savefig('/ismhome/staff/womo1998/Projects/2018_ISMA/paper/figures/cov_freq_vs_data_qual.pdf', dpi=300)
                plt.close('all')
            else:
                plt.show()        
    
    # cov_freq over kurtosis
    if ck:
        print_context_dict['figure.figsize']=(5.53*0.5,2.96)
        with matplotlib.rc_context(rc=print_context_dict):
            fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
            quantity='strain_rosettes'
            mode=0
            for i,(color,quantity) in enumerate(zip(['#006b9455','maroon'],['accel','strain_rosettes'])):

                q_1=['cov_frequencies']
                q_2=['kurtosis_m']

                postprocess_modal_results(quantity, duration, filter_errors=False, 
                                          wind_range=0, temp_range=0, mode_pair=mode, 
                                          q_1=q_1, q_2=q_2, 
                                          fig=fig,axes=axes[i], color=color,scatter=False,
                                          hexbin_extent = [-0.5,10,-4.5,0], log_scale='y')

                mode_str = ['all','first','second','third', 'fourth', 'fifth'][mode]
                axes[i].set_ylabel('')

                plt.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.14, hspace=0.04)
                axes[i].yaxis.set_ticks_position('left')
            axes[0].set_yscale('log')
            axes[0].set_ylim((10**-4.5,10**0))
            axes[0].set_xlim((-0.5,10))
            
            axes[0].yaxis.set_visible(True)
            axes[0].yaxis.set_ticks_position('left')

            axes[1].yaxis.set_visible(True)
            axes[1].yaxis.set_ticks_position('left')
            fig.text(0.01, 0.55, '$c_{v_f} [-]$', va='center', rotation='vertical')
            
            if save_figures:
                plt.gcf().savefig(figpath + 'cov_freq_vs_kurtosis_pres.png', dpi=300)
                #plt.gcf().savefig('/ismhome/staff/womo1998/Projects/2018_ISMA/paper/figures/cov_freq_vs_kurtosis.pdf', dpi=300)
                plt.close('all')
            else:
                plt.show()
    
    # main_dir over wind_dir
    if dd and quantity == 'accel':
        # print_context_dict['figure.figsize']=(5.53*0.62,2.96)
        with matplotlib.rc_context(rc=print_context_dict):
            fig,axes = modal_dirs(path, quantity, update=True)
            #plt.figure(fig)
            fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.14, hspace=0.04, wspace=0.04)
            fig.text(0.01, 0.55, 'Modeshape Direction [\si{\degree}]', va='center', rotation='vertical')
            
            axes[1,0].set_ylim((-45,315))
            axes[1,0].set_yticks([-45,0,45,90,135,180,225,270,315,])
            axes[1,0].set_yticklabels(['','E','','S','','W','','N',''])
            axes[1,0].set_xlim((90,360+90))
            axes[1,0].set_xticks([90,135,180,225,270,315,360,405,450])
            axes[1,0].set_xticklabels(['E','','S','','W','','N','','E'])

#             axes[0,0].set_yticks([0,45,90,135,180,225,270,315,360])
#             axes[0,0].set_yticklabels(['E','S','','W','','N','','E'])
            axes[0,0].set_xticks([])

            
            axes[0,1].set_xticks([])
            axes[1,1].set_xticks([])
            #plt.show()
            
            if save_figures:
                plt.gcf().savefig(figpath + f'q_{quantity}-d_{minutes}-directions_msh_vs_wind.png', dpi=300)
                #plt.gcf().savefig('/ismhome/staff/womo1998/Projects/2018_ISMA/paper/figures/main_dir_vs_wind.pdf', dpi=300)
                plt.close('all')
            else:
                plt.show()
    
    return
    
                    
if __name__ == '__main__':
    # main()
    test()
