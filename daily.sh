#!/bin/bash

PYTHONPATH=/vegas/users/staff/womo1998/git/pyOMA:/vegas/users/staff/womo1998/Projects/2015_modal_analysis_tower/strain_conversion
export PYTHONPATH
cd /vegas/users/staff/womo1998/Projects/2015_modal_analysis_tower/code/

/vegas/apps/compiler/intel/intelpython3/bin/python daily.py 1 1 1 0 | mail -s "Structural Monitoring Tower: Daily Analysis of Acceleration Records" volkmar.zabel@uni-weimar.de
/vegas/apps/compiler/intel/intelpython3/bin/python daily.py 1 1 1 1 | mail -s "Structural Monitoring Tower: Daily Analysis of Wind Records" volkmar.zabel@uni-weimar.de
/vegas/apps/compiler/intel/intelpython3/bin/python daily.py 1 1 1 2 | mail -s "Structural Monitoring Tower: Daily Analysis of Temperature Records" volkmar.zabel@uni-weimar.de
# /vegas/apps/compiler/intel/intelpython3/bin/python daily.py 1 1 1 3 | mail -s "Structural Monitoring Tower: Daily Analysis of Strain Records" volkmar.zabel@uni-weimar.de

# arg[1]: worker number (stats, modal), arg[2]: number of workers, arg[3]: duration (10,30,60,120 minutes), arg[4]: quantity (accel, wind,temp, strain)
