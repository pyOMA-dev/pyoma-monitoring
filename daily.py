"""CLI entry-point for the tower monitoring pipeline.

Run the file-info scan, statistics computation, OMA, and/or plotting for a
given measurement quantity and look-back duration. Intended to be called from
the cron wrapper ``daily2.sh`` but can also be invoked directly for ad-hoc
analysis::

    python daily.py -d 120 -q accel --file_info --stats --modal --plot \\
        --tmp_dir=/tmp/shm
"""
import sys
import getopt
import numpy as np
import pandas as pd
import site_tower  # registers and activates the tower site in the monitoring engine  # noqa: F401
from monitoring import get_file_info, get_file_list, round_dt, get_stats, get_modal_results
import config
from post_processing import plot_daily, plot_waterfall
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# TODO:
# - store output of file_info, stats, modal, etc. as a log file and attach it to mail
# - only log exit status and other relevant information, e.g. num_file, num_slices, dtstart, runtimes, statistics, etc. to stdout

def main(argv):
    
    opts, _args = getopt.getopt(argv,'hd:q:v',['file_info', 'stats', 'modal', 'plot','tmp_dir=', 'dtstart=','loglevel='])
    
    duration = None
    quantity = None
    file_info = False
    stats = False
    modal = False
    plot = False
    dtstart = None
    tmp_dir = None
    
    for opt, arg in opts:
        if opt == '-h':
            print ('daily.py -d <duration in minutes> -q <quantity> --file_info --stats --modal --plot --tmp_dir=<directory path> --dtstart=YYYY-MM-DD hh:mm --loglevel=INFO')
            sys.exit()
        if opt == '-d':
            duration = int(arg)
            if duration not in [10,30,60,120]:
                raise ValueError(f"Duration {duration} is not supported. Choices [10, 30, 60, 120]")
            duration = pd.Timedelta(minutes=duration)
        if opt == '-q':
            quantity = arg
            if quantity not in ['accel', 'wind', 'temp', 'strain_rosettes']:
                raise ValueError(f'Quantity "{quantity}" is not supported. Choices [accel, wind, temp, strain_rosettes]')
        if opt == '--file_info':
            file_info = True
        if opt == '--stats':
            stats = True
        if opt == '--modal':
            modal = True
            if quantity not in ['accel', 'strain_rosettes']:
                raise ValueError(f'Modal analysis can only be performed on vibration data ("accel", "strain_rosettes"')
        if opt=='--loglevel':
            logger.setLevel(arg)
            for name in logging.root.manager.loggerDict:  # pylint: disable=no-member
                logging.getLogger(name).setLevel(arg)
        if opt == '--plot':
            plot = True
        if opt == '--tmp_dir':
            tmp_dir = arg
            assert os.path.exists(tmp_dir)
        if opt == '--dtstart':
            dtstart = np.datetime64(arg)
                
    if plot:
        if tmp_dir is None:
            plot_dir = os.getcwd()
        else:
            plot_dir = tmp_dir 
            
    if duration is None or quantity is None:
        print ('daily.py -d <duration in minutes> -q <quantity> --file_info --stats --modal --plot --tmp_dir=<directory path> --dtstart=YYYY-MM-DD hh:mm --loglevel=INFO')
        sys.exit()
    minutes = int(duration.total_seconds()/60)
    db_path = os.path.join(config.db_root_path, f'{minutes}-minutes/')
    # path='/vegas/scratch/womo1998/towerdata/{}-minutes/'.format(duration)
        
    _subpath = config.subpaths[quantity]
    origin = config.origins[quantity]
    
    start_up_str = 'Tower Monitoring System - Commandline Tool\n\n'
    start_up_str += 'Selected parameters:\n'
    start_up_str += f'Quantity: \t\t {quantity}\n'
    start_up_str += f'Duration: \t\t {minutes} minutes\n'
    start_up_str += f'Results stored at: \t {db_path}\n'
    if plot:
        start_up_str += f'Figures stored at: \t {plot_dir}\n'
    start_up_str += '\nSelected analyses:\n'
    
    if file_info:
        start_up_str += ' - Read in newly transfered files\n'
    if stats:
        if file_info:
            start_up_str += ' - Do statistical analysis on new signal slices\n'
        if dtstart is not None:
            start_up_str += f' - Do statistical analysis on all signal slices since {dtstart}\n'
        else:
            start_up_str += ' - Do statistical analysis on all signal slices that have not yet been analysed\n'
    if modal:
        if file_info:
            start_up_str += ' - Do modal analysis on new signal slices\n'
        if dtstart is not None:
            start_up_str += f' - Do modal analysis on all signal slices since {dtstart}\n'
        else:
            start_up_str += ' - Do modal analysis on all signal slices that have not yet been analysed\n'
    
    for line in start_up_str.split('\n'):
        logger.info(line)

    if dtstart is None and not file_info:
        raise RuntimeError('dtstart must be provided if file_info shall not be updated')
    
    if file_info:
        
        fi_ds = get_file_info(origin, create_new=False)
        
        filtered_list = get_file_list(origin, True, fi_ds)
        # empty list -> no new files arrived
        if not filtered_list and dtstart is None:
            logger.warning("No new files have arrived since the last run. Check that the monitoring system is online and working!")
            return
        
        fi_ds = get_file_info(origin, create_new=True, skip_existing=True, reduced=True, filtered_list=True)
        
        if dtstart is None:
            # get start of current file list, to only create statistics for these new files
            # if script breaks before processing all files for stats, in the next run this function will be inconsistent 
            filename_list = [os.path.basename(file) for file in filtered_list]
            
            dset = fi_ds.file_name.isin(filename_list)
            start_times =fi_ds.start_time[dset].astype('datetime64[s]')
            if len(start_times) > 0:
                dtstart = start_times.min().values
            else:
                logger.warning('Processing files has failed. Check analysis scripts and file integrity.')
                return
    else:
        fi_ds = get_file_info(origin, create_new=False)
    
    if dtstart is not None and tmp_dir is not None:
        with open(os.path.join(tmp_dir,'dtstart.tmp'),'wt') as f:
            f.write(str(dtstart))
    
    if stats:
        # adjust dtstart to time_iterator slices
        dtstart = round_dt(dtstart, duration.to_timedelta64(), floor=True)
        stats_ds = get_stats(quantity, duration, 
                             fi_ds, dtstart=dtstart, 
                             create_new=True, skip_existing=True, chunksize=500)#, until=until)
    else:
        stats_ds = get_stats(quantity, duration)
    
    if modal: 
        _modal_ds = get_modal_results(quantity, duration, stats_ds,
                                  skip_existing=True, create_new=True, 
                                  filter_errors=False, 
                                  chunksize=50, missing=True)
        
    if plot:
        fig1, fig2 = plot_daily(quantity, duration, dtstart)
        fig1.savefig(os.path.join(plot_dir,f'stats_{quantity}_{minutes}.png'))
        if modal and fig2 is not None:
            fig2.savefig(os.path.join(plot_dir,f'modal_{quantity}_{minutes}.png'))
            fig3 = plot_waterfall(quantity, duration, dtstart)
            fig3.savefig(os.path.join(plot_dir,f'spec_{quantity}_{minutes}.png'))
            
            
    
    
if __name__ == '__main__':
    
    main(sys.argv[1:])
