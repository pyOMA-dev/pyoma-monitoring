#!/bin/bash

#rsync -a srv-grk:/home/towermonitoring/towerdata/ /vegas/scratch/womo1998/towerdata/towerdata_bin
#module load python/intelpython3.7

cd /home/towermonitoring/analysis/code/

PATH=/vegas/apps/compiler/intel/intelpython3.7/bin:$PATH
export PATH
PYTHONPATH=/home/towermonitoring/analysis/pyOMA
export PYTHONPATH

# Configure the recipient(s) before use. Multiple addresses can be
# space-separated; whether that is supported depends on your mail(1) version.
RECIPIENTS="your@email.de"

TMPDIR=/dev/shm/tower_tmp
rm -rf ${TMPDIR}

mkdir ${TMPDIR}
python daily.py -d 120 -q temp --file_info --stats --tmp_dir=${TMPDIR}> ${TMPDIR}/tower_out.txt
if test -f "${TMPDIR}/dtstart.tmp"; then
	DTSTART=`cat ${TMPDIR}/dtstart.tmp`
	python daily.py -d 60 -q temp --stats --tmp_dir=${TMPDIR} --dtstart=${DTSTART} >> ${TMPDIR}/tower_out.txt
	python daily.py -d 30 -q temp --stats --tmp_dir=${TMPDIR} --dtstart=${DTSTART} >> ${TMPDIR}/tower_out.txt
	python daily.py -d 10 -q temp  --stats --plot --tmp_dir=${TMPDIR} --dtstart=${DTSTART}  >> ${TMPDIR}/tower_out.txt
	cat ${TMPDIR}/tower_out.txt | mail -s "Structural Monitoring Tower: Daily Analysis of Temperature Records" -a ${TMPDIR}/stats_temp_10.png ${RECIPIENTS}
else
	cat ${TMPDIR}/tower_out.txt | mail -s "Structural Monitoring Tower: Daily Analysis of Temperature Records" ${RECIPIENTS}
fi
rm -rf ${TMPDIR}

mkdir ${TMPDIR}
python daily.py -d 120 -q wind --file_info --stats --tmp_dir=${TMPDIR}> ${TMPDIR}/tower_out.txt
if test -f "${TMPDIR}/dtstart.tmp"; then
	DTSTART=`cat ${TMPDIR}/dtstart.tmp`
	python daily.py -d 60 -q wind --stats --tmp_dir=${TMPDIR} --dtstart=${DTSTART} >> ${TMPDIR}/tower_out.txt
	python daily.py -d 30 -q wind --stats --tmp_dir=${TMPDIR} --dtstart=${DTSTART} >> ${TMPDIR}/tower_out.txt
	python daily.py -d 10 -q wind  --stats --plot --tmp_dir=${TMPDIR} --dtstart=${DTSTART}  >> ${TMPDIR}/tower_out.txt
	cat ${TMPDIR}/tower_out.txt | mail -s "Structural Monitoring Tower: Daily Analysis of Wind Records" -a ${TMPDIR}/stats_wind_10.png ${RECIPIENTS}
else
	cat ${TMPDIR}/tower_out.txt | mail -s "Structural Monitoring Tower: Daily Analysis of Wind Records" ${RECIPIENTS}
fi
rm -rf ${TMPDIR}

mkdir ${TMPDIR}
python daily.py -d 120 -q accel --file_info --stats --modal --tmp_dir=${TMPDIR}> ${TMPDIR}/tower_out.txt
if test -f "${TMPDIR}/dtstart.tmp"; then
	DTSTART=`cat ${TMPDIR}/dtstart.tmp`
	python daily.py -d 60 -q accel --stats --modal --tmp_dir=${TMPDIR} --dtstart=${DTSTART} >> ${TMPDIR}/tower_out.txt
	python daily.py -d 30 -q accel --stats --modal --tmp_dir=${TMPDIR} --dtstart=${DTSTART} >> ${TMPDIR}/tower_out.txt
	python daily.py -d 10 -q accel  --stats --plot --modal --tmp_dir=${TMPDIR} --dtstart=${DTSTART}  >> ${TMPDIR}/tower_out.txt
	cat ${TMPDIR}/tower_out.txt | mail -s "Structural Monitoring Tower: Daily Analysis of Acceleration Records" -a ${TMPDIR}/stats_accel_10.png -a ${TMPDIR}/modal_accel_10.png -a ${TMPDIR}/spec_accel_10.png ${RECIPIENTS}
else
	cat ${TMPDIR}/tower_out.txt | mail -s "Structural Monitoring Tower: Daily Analysis of Acceleration Records" ${RECIPIENTS}
fi
rm -rf ${TMPDIR}
