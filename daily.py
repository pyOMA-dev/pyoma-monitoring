import sys
import getopt
import numpy as np
import pandas as pd
from main_v2 import get_file_info, get_file_list,round_dt, get_stats,get_modal_results
import config
from post_processing import plot_daily
import os
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main(argv):
    
    opts, args = getopt.getopt(argv,'hd:q:v',['file_info', 'stats', 'modal', 'plot_dir=', 'dtstart=','loglevel='])
    
    duration = None
    quantity = None
    file_info = False
    stats = False
    modal = False
    plots = False
    dtstart = None
    
    for opt, arg in opts:
        if opt == '-h':
            print ('daily.py -d <duration in minutes> -q <quantity> --file_info --stats --modal --plot_dir --dtstart=YYYY-MM-DD hh:mm --loglevel=INFO')
            sys.exit()
        if opt == '-d':
            duration = int(arg)
            if duration not in [10,30,60,120]:
                raise ValueError(f"Duration {duration} is not supported. Choices [10, 30, 60, 120]")
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
        if opt == '--plot_dir':
            plots = True
            plot_dir = arg
            assert os.path.exists(plot_dir)
        if opt == '--dtstart':
            dtstart = np.datetime64(arg)
        
    if duration is None or quantity is None:
        print ('daily.py -d <duration in minutes> -q <quantity> --file_info --stats --modal --plot_dir --dtstart=YYYY-MM-DD hh:mm --loglevel=INFO')
        sys.exit()
        
    db_path = os.path.join(config.db_root_path, f'{duration}-minutes/')
    # path='/vegas/scratch/womo1998/towerdata/{}-minutes/'.format(duration)
        
    subpath = config.subpaths[quantity]
    origin = config.origins[quantity]
    
    start_up_str = 'Geyer Monitoring System - Commandline Tool\n\n'
    start_up_str += 'Selected parameters:\n'
    start_up_str += f'Quantity: \t\t {quantity}\n'
    start_up_str += f'Duration: \t\t {duration} minutes\n'
    start_up_str += f'Results stored at: \t {db_path}\n'
    if plots:
        start_up_str += f'Figures stored at: \t {plot_dir}\n'
    start_up_str += '\nSelected analyses:\n'
    
    if file_info:
        start_up_str += ' - Read in newly transfered files\n'
    if stats:
        if file_info:
            start_up_str += ' - Do statistical analysis on new signal slices\n'
        else:
            start_up_str += ' - Do statistical analysis on all signal slices that have not yet been analysed\n'
    if modal:
        if file_info:
            start_up_str += ' - Do modal analysis on new signal slices\n'
        else:
            start_up_str += ' - Do modal analysis on all signal slices that have not yet been analysed\n'
    
    for line in start_up_str.split('\n'):
        logger.info(line)

    if dtstart is None and not file_info:
        raise RuntimeError('dtstart must be provided if file_info shall not be updated')
    
    if file_info:
        fi_ds = get_file_info(db_path, origin, create_new=False)
        
        filtered_list = get_file_list(origin, True, fi_ds)
        # empty list -> no new files arrived
        if not filtered_list:
            logger.warning("No new files have arrived since the last run. Check that the monitoring system is online and working!")
            return
        
        fi_ds = get_file_info(db_path, origin, create_new=True, skip_existing=True, reduced=True, filtered_list=True)
        
        if dtstart is None:
            # get start of current file list, to only create statistics for these new files
            # if script breaks before processing all files for stats, in the next run this function will be inconsistent 
            filename_list = [os.path.basename(file) for file in filtered_list]
            
            dset = fi_ds.file_name.isin(filename_list)
            start_times =fi_ds.start_time[dset].astype('datetime64[s]')
            if len(start_times) > 0:
                dtstart = start_times.min().values
                os.environ['DTSTART'] = repr(dtstart)
            else:
                logger.warning('Processing files has failed. Check analysis scripts and file integrity.')
                return
    else:
        fi_ds = get_file_info(db_path, origin, create_new=False)
        
    if stats:
        # adjust dtstart to time_iterator slices
        dtstart = round_dt(dtstart, duration, floor=True)
        stats_ds = get_stats(db_path, quantity, fi_ds, 
                             duration = pd.Timedelta(minutes=duration), dtstart=dtstart, 
                             create_new=True, skip_existing=True, chunksize=500)#, until=until)
    else:
        stats_ds = get_stats(db_path, quantity)
    
    if modal: 
        modal_ds = get_modal_results(db_path, quantity, stats_ds, 
                                  skip_existing=True, create_new=True, 
                                  filter_errors=False, 
                                  chunksize=50, missing=True)
        
    if plots:
        fig1, fig2 = plot_daily(db_path, quantity, duration, dtstart)
        fig1.savefig(os.path.join(plot_dir,f'stats_{quantity}_{duration}.png'))
        if modal:
            fig2.savefig(os.path.join(plot_dir,f'modal_{quantity}_{duration}.png'))
    
    
if __name__ == '__main__':
    
    main(sys.argv[1:])