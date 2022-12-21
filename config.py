import os
import datetime
from collections import deque
file_cache = deque(maxlen=25)

origins = {'accel':'accel', 
          'wind':'wind', 
          'temp':'temp', 
          'strain_rosettes':'strain', 
          'strain_bolts':'strain'}

ds_cache = {}
for origin in origins.values():
    ds_cache[f'{origin}_file_info']={'ds':None, 'mtime':None}
for quantity in origins.keys():
    ds_cache[f'{quantity}_stats']={'ds':None, 'mtime':None}
    if quantity in ['accel', 'strain_rosettes']:
        ds_cache[f'{quantity}_modal']={'ds':None, 'mtime':None}
        
pid=str(os.getpid())

if os.uname()[1]=='srv-grk':     
    subpaths = {'accel':'towerdata',
           'wind':'towerdata',
           'temp':'towerdata',
           'strain':'strain_data'}
    file_root_path = '/home/towermonitoring/'
    slice_root_path = '/vegas/scratch/womo1998/towerdata/'
    db_root_path = '/home/towermonitoring/analysis/result_db/'
    modal_conf_dir = '/srv-grk/towermonitoring/analysis/modal_source_files/'
else:
    subpaths = {'accel':'towerdata_bin',
           'wind':'towerdata_bin',
           'temp':'towerdata_bin',
           'strain':'strain_data_bin'}
    file_root_path = '/vegas/scratch/womo1998/towerdata/'
    slice_root_path = '/vegas/scratch/womo1998/towerdata/'
    db_root_path = '/vegas/scratch/womo1998/towerdata/result_db/'
    modal_conf_dir = '/vegas/scratch/womo1998/towerdata/modal_source_files/'

dtstarts = {'accel':datetime.datetime(2015,5,20),
            'wind':datetime.datetime(2015,5,20),
            'temp':datetime.datetime(2015,5,20),
            'strain':datetime.datetime(2016,12,16)}

all_channels = {
    'accel' : ['Accel_01','Accel_02',],
    'wind' : ['Wr', 'Wg',],
    'strain_rosettes' : ['A_Temp','B_Temp','C_Temp','D_Temp_1','D_Temp_2',
                           'D_Temp_3','A_z','A_t','A_zt','B_z','B_t',
                           'B_zt','C_z','C_t','C_zt','D_z','D_t','D_zt',],
    'strain_bolts' : ['10_Temp','10_z1','10_z2','8_z1','8_z2',
                        '8_z3','9_z1','9_z2','9_z3',],
    'temp' : ['Pt100_01','Pt100_02','Pt100_03','Pt100_04',
                       'Pt100_05']
    }

optional_channels = {'accel': ['Accel_01_top', 'Accel_02_top', 
                               'Accel_03_top', 'Accel_04_top',
                               'Accel_03','Accel_04', 'Accel_05',
                               'Accel_06','Accel_07','Accel_08'] ,
                    'wind': ['Wr_top','Wg_top']}

strain_channels = ['A_z','A_t','A_zt','B_z','B_t','B_zt','C_z','C_t',
                       'C_zt','D_z','D_t','D_zt','10_z1','10_z2','8_z1',
                       '8_z2','8_z3','9_z1','9_z2','9_z3',]

temp_channels = ['10_Temp', 'D_Temp_1','D_Temp_2','D_Temp_3','C_Temp','B_Temp','A_Temp']

ranges={'Accel_01':(-5,5), 'Accel_01_top':(-1,1), 'Accel_02':(-5,5),
        'Accel_02_top':(-1,1), 'Accel_03':(-5,5), 'Accel_03_top':(-1,1), 
        'Accel_04':(-5,5), 'Accel_04_top':(-1,1), 'Accel_05':(-1,1), 
        'Accel_06':(-1,1), 'Accel_07':(-10,10), 'Accel_08':(-10,10), 
        'Tagessekunden':(0,86400),  'Time':(0,86400), 'Wg':(-1,61), 
        'Wg_top':(-1,61),  'Wr':(-185,185), 'Wr_top':(-185,185),
        'Wx':(-61,61), 'Wx_top':(-61,61), 'Wy':(-61,61), 'Wy_top':(-61,61),
        'Pt100_01':(-40,80), 'Pt100_02':(-40,80), 'Pt100_03':(-40,80),
        'Pt100_04':(-40,80), 'Pt100_05':(-40,80), 'Pt100_top':(-40,80),
        'Pt100_box':(-40,80), 'WTemp':(-40,80), 'WTemp_top':(-40,80),
        'D_Temp_1':(-40,60), 'D_Temp_2':(-40,60), 'D_Temp_3':(-40,60),
        'A_Temp':(-40,60), 'B_Temp':(-40,60), 'C_Temp':(-40,60),
        'A_z':(-0.0007,0.0002), 'A_t':(-0.0005,0.0005), 'A_zt':(-0.0006,0.0000),
        'B_z':(-0.0002,0.0002), 'B_t':(-0.0004,0.0000), 'B_zt':(-0.0003,0.0001),
        'C_z':(-0.0003,0.0001), 'C_t':(-0.0002,0.0002), 'C_zt':(-0.0004,0.0000),
        'D_z':(-0.0002,0.0002), 'D_t':(-0.0002,0.0002), 'D_zt':(-0.0003,0.0000)
        }
initial_wl ={
        '10_Temp':1520.1, '10_z1':1527.59, '10_z2':1535.07,
        '8_z1':1565.06, '8_z2':1572.44, '8_z3':1579.84,
        '9_z1':1565, '9_z2':1572.5, '9_z3':1580,
        'D_Temp_1':1520, 'D_Temp_2':1527.5, 'D_Temp_3':1535,
        'D_z':1542.5, 'D_t':1550, 'D_zt':1557.5,
        'C_z':1565, 'C_t':1572.5, 'C_zt':1580,
        'C_Temp':1520.28, 'B_Temp':1530.14, 'A_Temp':1539.97,
        'A_z':1542.5, 'A_t':1550, 'A_zt':1557.5,
        'B_z':1565, 'B_t':1572.5, 'B_zt':1580,
        }