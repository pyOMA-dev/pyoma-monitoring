"""Low-level reader for LabVIEW binary files produced by the Geyer mast DAQ.

Parses the proprietary binary format written by LabVIEW into numpy arrays,
detects missing peaks (marked as NaN), and returns per-channel time-series
data together with header metadata and sample-rate information.

Authors: Heiko Beinersdorf, Simon Marwitz (Feb 2017, Aug 2017)
"""


import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import bz2


import datetime,time, pytz
import struct
import numpy as np

#import codecs
import pandas as pd
import pandas.errors 

import warnings
import matplotlib.pyplot as plot




berlin_dst=pytz.timezone('Europe/Berlin') # cet/cest
cet = pytz.FixedOffset(offset=+60)
utc = pytz.UTC

warnings.simplefilter('always', UserWarning)


peaksExpected = 9
wlExpected = [np.array([[1520,1527.5,1535,1542.5,1550,1557.5,1565,1572.5,1580]]),
              np.array([[1520,1527.5,1535,1542.5,1550,1557.5,1565,1572.5,1580]]),
              np.array([[1520,1527.5,1535,1542.5,1550,1557.5,1565,1572.5,1580]]),
              np.array([[1520,1530,1540,1542.5,1550,1557.5,1565,1572.5,1580]]),]

wlThreshold = 1.2 # threshold for assignment measured wavelengths to expected wavelengths

namesChannels = [['12_z1','12_z2','12_z3','13_z1','13_z2','13_z3','15_Temp','15_z1','15_z2',],
                ['10_Temp','10_z1','10_z2','8_z1','8_z2','8_z3','9_z1','9_z2','9_z3'],
                ['D_Temp_1','D_Temp_2','D_Temp_3','D_z','D_t','D_zt','C_z','C_t','C_zt'],
                ['C_Temp','B_Temp','A_Temp','A_z','A_t','A_zt','B_z','B_t','B_zt']]

# FUNCTION: convert labview timestamp to python date object
def timestamp_labview2python(bTimeStamp):
    if bTimeStamp:
        # GET labview timestamp (128bit = i64 (seconds) + u64 (partition of second))
        labview_timestamp=struct.unpack('>qQ', bTimeStamp)
        # CONVERT timestamp to python date object (Labview Timestamp: seconds since 01.01.1904 UTC)
        d = datetime.datetime.strptime("01-01-1904", "%m-%d-%Y") + datetime.timedelta(seconds=labview_timestamp[0] + labview_timestamp[1]/0xffffffffffffffff) # ADD seconds
        return d # in UTC

def read_bin(inFileName, stats_only=False, wavepower=False, indices_only=False, path=None):
    
    if isinstance(inFileName, str):
        bfile = open(inFileName , 'rb')
        path=inFileName
    else:
        bfile=inFileName
    
    # GET file size
    bfile.seek(0, 2)
    bSize = bfile.tell()
    bfile.seek(0)
    
    # GET labview timestamp (128bit)
    startTimestamp=timestamp_labview2python(bfile.read(16))
    # startTimestamp and endTimestamp of previous file are generated simultaneously i.e. are equal
    # file change occurs after num_lines are written to file, that means if
    # num_lines is not divisible by the number of channels, the file change occurs between channel switches
    # interleaving of files must be done subsequently
    
    # GET number of samples
    samples=(struct.unpack('>h', bfile.read(2)))[0]
    
    if stats_only:
        return [], [], localize(startTimestamp), samples/4, np.zeros((0,0))
    
    # SET IDs & counter
    lastIndex = None
    firstIndex = None
    firstChannel = None

    lastChannel = -1
    
    channelData = [[],[],[],[]]
    
    a = np.empty((1,peaksExpected))
    a[:] = np.NAN
    
    if wavepower:
        channelDataPow = [[],[],[],[]]
        
        b = np.empty((1,peaksExpected))
        b[:] = np.NAN
        
    
    # LOOP over all data lines
    iChannel = 0
    iIndex = 0

            
    while True :
        
        # GET bytes to read
        rest=bSize-bfile.tell()
    
        # TEST eof (eof = 128bit timestamp)
        if rest == 16:
            endTimestamp=timestamp_labview2python(bfile.read(16))
            break #right EOF
        elif rest < 16:
            if firstIndex is not None:
                endTimestamp=startTimestamp+datetime.timedelta(seconds=(iIndex-firstIndex)/samples)+datetime.timedelta(seconds=1/(samples))
            else:
                endTimestamp=startTimestamp
            break
        
        # GET number of samples
        (iChannel,iIndex,nPeaks)=struct.unpack('>BIB', bfile.read(6))
        #print(iChannel,iIndex)

       
        iChannel -= 1 # binary channels start at 1, python "channels" start at 0
        

        if lastIndex is None:
            lastIndex = iIndex
            firstIndex = iIndex
            firstChannel = iChannel
            
        if indices_only:
            bfile.read(2*(4 + nPeaks * 8))
            continue   
         
        #ind_gap = False
        # if indices are missing fill with nan at rotating channels
        while iIndex > lastIndex + 1:# index counts up with channels
            #ind_gap = True 
            logger.debug("Index Gap: {0} -> {1}".format(lastIndex,iIndex))
            #print(lastChannel+1, [len(channel) for channel in channelData])

            channelData[lastChannel+1].append(np.copy(a))
            
            if wavepower:
                channelDataPow[lastChannel+1].append(np.copy(b))
                
            if lastChannel + 1 < 3:
                lastChannel += 1
            else:
                lastChannel = -1
                
            lastIndex += 1  
            
        #if ind_gap:
        #   print(iChannel, lastChannel+1, iIndex, lastIndex, [len(channel) for channel in channelData])
            
        # if measurement starts at a different channel than the first fill up preceding channels with nan
        while lastChannel < iChannel - 1:
            lastChannel += 1
            logger.debug('filling up channel {}  at first index {}'.format(lastChannel, iIndex))
            channelData[lastChannel].append(np.copy(a))
            if wavepower:
                channelDataPow[lastChannel].append(np.copy(b))
            
        #print(lastChannel, iChannel)
        #if iChannel != lastChannel + 1:
        #    print('error')
        
        assert iChannel == lastChannel + 1
        
        #assign iChannel to lastChannel for next iteration
        # in next iteration lastChannel+1 should be equal to iChannel
        # therefore set lastChannel to -1 if iChannel==3 
        if iChannel < 3:
            lastChannel = iChannel
        else:
            lastChannel = -1
        
        # assing iIndex to lastIndex for next iteration
        lastIndex = iIndex
        
        # if no peaks were detected continue to next row/index/time_instant
        if nPeaks == 0: 
            bfile.read(8)
            channelData[iChannel].append(np.copy(a))
            if wavepower:
                channelDataPow[iChannel].append(np.copy(b))
            continue

        # GET array length and arrays
        PeakWL = (struct.unpack('>I'+str(nPeaks)+'d', bfile.read(4 + nPeaks * 8)))[1:]
        PeakPower = (struct.unpack('>I'+str(nPeaks)+'d', bfile.read(4 + nPeaks * 8)))[1:]
        
        # sort measured waveLengths according to the predefined (expected) wavelenghts
        distance = np.ma.array(wlExpected[iChannel].T - np.array(PeakWL))
        distance = np.abs(distance)
        distance[distance>wlThreshold] = np.ma.masked
        
        a_ = np.copy(a)
        if wavepower:
            b_ = np.copy(b)
        while not np.all(distance.mask):
            
            exp_ind,this_ind = np.unravel_index(np.ma.argmin(distance), distance.shape)

            a_[0,exp_ind]=PeakWL[this_ind]
            if wavepower:
                b_[0,exp_ind]=PeakPower[this_ind]
            
            distance[exp_ind,:]=np.ma.masked
            distance[:,this_ind]=np.ma.masked
        
        channelData[iChannel].append(a_)
        
        if wavepower:
            channelDataPow[iChannel].append(b_)
            
    if indices_only:
        return localize(startTimestamp), utc.localize(endTimestamp), firstChannel, iChannel, firstIndex, iIndex  # pylint: disable=no-value-for-parameter
      
    # if file ends with another channel than the last, fill up with nans
    iChannel += 1
    while iChannel <=3:
        logger.debug('filling up channel {}  at last index {}'.format(iChannel,iIndex))
        channelData[iChannel].append(np.copy(a))
        
        if wavepower:
            channelDataPow[iChannel].append(np.copy(b))
        iChannel += 1
    
    # generate full numpy array and headers
    arrays=[]
    headers=[]
    for channel, channel_data in enumerate(channelData):
        if channel == 0:
            continue#skip the first row for Geyer Monitoring system
        arrays.append(np.vstack(channel_data))
        headers += namesChannels[channel]
    
    a = np.hstack(arrays)
    if wavepower:
        arraysPow=[]
        headersPow=[]
        for channel, channel_data in enumerate(channelDataPow):
            if channel == 0:
                continue#skip the first row for Geyer Monitoring system
            arraysPow.append(np.vstack(channel_data))
            headersPow += namesChannels[channel]
        #print([len(abc) for abc   in arraysPow])
        aPow = np.hstack(arraysPow)

    
    if wavepower:
        
        return headers+headersPow, ['nm' for header in headers]+['?' for header in headers], localize(startTimestamp), samples/4, np.hstack((a,aPow))
    else:
        if path is not None:
            np.savez(path.rstrip('.bz2')+'.npz',headers=headers, units=['nm' for header in headers], startTimestamp=localize(startTimestamp), sample_rate=samples/4, measurement=a)
        
        return headers, ['nm' for header in headers], localize(startTimestamp), samples/4, a
    
def localize(startTimestamp):
    '''
    the start time, e.g. the time stamp that is written down inside the file 
    is recorded in local time with daylight saving, if turned on
    on 2016-12-15 strain recordings started, labview timestamps are in UTC
    on 2017-10-29 last daylight savings clock change occured, since then PC was running on CET 
    on 2018-01-15 illumisense recordings started, timestamps are in local time, i.e. CET
    on 2019-4-25 pc was set to UTC, timestamps are in CET
    
    2016-12-15 - 2017-10-29 startTimestamp in 'Europe/Berlin' with DST (CET/CEST)
    2017-10-29 - 2019-04-25 startTimestamp in CET (without DST)
    2019-04-25 - ...        startTimestamp in UTC
    
    print(returned_timestamp.astimezone(pytz.utc))
    '''
    assert not startTimestamp.tzinfo
    if startTimestamp < datetime.datetime(2018,1,15): # labview recording
        return utc.localize(startTimestamp)  # pylint: disable=no-value-for-parameter
    else: # illumisense recording
        if startTimestamp <= datetime.datetime(2017,10,29,1,59,59): # just the instant before clock change
            return berlin_dst.localize(startTimestamp)  # pylint: disable=no-value-for-parameter
        elif startTimestamp < datetime.datetime(2019,4,25):
            return cet.localize(startTimestamp)  # pylint: disable=no-value-for-parameter
        else:
            return utc.localize(startTimestamp)  # pylint: disable=no-value-for-parameter
    
def manipulate_data(measurement, start_time, sample_rate, previous_a=None, previous_delta=None, previous_start_time=None):
    
    import scipy.interpolate
    from operator import itemgetter
    from itertools import groupby
    
    # some files show jumps between certain wavelengths, an educated guess
    # is, that the quality of the bond is not good enought, s.t. there are
    # inhomogenous strains in the fibre-bragg-grating. this leads to widening
    # of the peaks and therefore the internal peak-picking algorithm jumps from 
    # one side of the peak to the other side
    # these jumps are manipulated in the sense, that consistent data is available
    # nevertheless strain levels are consequently biased
    #
    # remove , manipulate offsets, interpolate missing values    
    
    # identify contiguos missing datapoints
    
    def manipulate_jumps():
            
        if plt:
            #fig,axes=plot.subplots(nrows=1, ncols=2, sharex=False, sharey=True,gridspec_kw={'width_ratios':[.8,.2]},tight_layout=1)#plot.figure()
            #print(fig)
            axes[0].plot(np.copy(measurement[:,i]), alpha=.3,color='orange')
        diff = measurement[1:,i]-measurement[:-1,i]
        if plt:
            #fig2,axes2=plot.subplots(nrows=1, ncols=2, sharex=False, sharey=True,gridspec_kw={'width_ratios':[.8,.2]},tight_layout=1)
            axes2[0].plot(np.abs(diff[~np.isnan(diff)]))
            
            #bins=np.arange(transition_threshold-0.0005,np.max(np.abs(diff))+0.0005,0.001)    
            #axes2[1].hist(np.abs(diff),orientation='horizontal', alpha=.3, bins=bins ,align='left')
        
        #STEP 1:
        ################################################################
        # manipulate jumps within a single file
        # more than one jump may occur in a file at high vibration levels or sharp temperature drops
        if np.any(np.abs(diff[~np.isnan(diff)])>transition_threshold/2):
            
            #b=measurement[:,i][~np.isnan(measurement[:,i])]
            
            # use a histogram-based approach to group the data into two clusters
            bins=np.arange(np.nanmin(measurement[:,i])-0.0005,np.nanmax(measurement[:,i])+0.0015,0.001)    
            hist,bins = np.histogram(a=measurement[:,i], bins=bins, range=(np.nanmin(measurement[:,i]),np.nanmax(measurement[:,i])))
            #print(hist)
            hist = hist > 0
            # there should be no left and right zeros, since we start at the minimum and maximum
            # find large cluster of zeros between outer clusters of ones
            ranges = np.where(np.diff(hist)!=0)[0]
            ranges= np.append(ranges, hist.shape[0])
            
            zero_clusters = []
            one_clusters =  []
            last_ind = 0
            for ind in ranges:
                
                if not np.any(hist[last_ind:ind+1]):#hist is all zero
                    # ignore first or last zero cluster or one-element clusters
                    if last_ind == 0 or ind == hist.shape[0] or ind-last_ind <= np.floor(transition_threshold/2/0.001)-1:
                        pass                     
                    else:
                        zero_clusters.append((last_ind, ind))
                else: # hist should be all ones
                    assert np.all(hist[last_ind:ind+1])
                    one_clusters.append((last_ind,ind))
                    
                last_ind = ind+1
                
            # move all values towards the lowest cluster
            # save delta of last value 
            acopy = np.copy(measurement[:,i])
            acopy = acopy[~np.isnan(acopy)]
            
            last_value = acopy[-1]            
            first_value = acopy[0]
            
            for zero_cluster in reversed(zero_clusters):
                edges = bins[zero_cluster,]
                #print(edges)
                with np.errstate(invalid='ignore'):#ignoring nans in measurement[:,i]
                    indices_new = measurement[:,i]>np.mean(edges) # all values that have to be manipulateed
                
                adiff = diff[indices_new[1:]+indices_new[:-1]]
                adiff = adiff[~np.isnan(adiff)]
                adiff = np.abs(adiff)
                adiff = adiff[adiff>transition_threshold/2]
                if not len(adiff):
                    continue
                delta = -1.0*(np.nanpercentile(adiff,q=[50],interpolation='lower')+0.0)
                
                if plt:
                    c=np.copy(measurement)
                    c[:,i][np.logical_not(indices_new)]=np.nan            
                    
                    d=np.copy(measurement)
                    d[:,i][indices_new]=np.nan
                    
                    c +=delta
                    
                    axes[0].plot(d[:,i], color='red', ls='none',alpha=.5,marker='x',markersize=4)            
                    axes[0].plot(c[:,i], color='blue',alpha=.5,marker='+',ls='none',markersize=4)
                    
                measurement[indices_new,i] += delta
                
                if plt:
                    axes[0].plot(np.copy(measurement[:,i]), alpha=.3,color='green')
                logger.debug('Channel \t {} \t manipulating jumps \t {}\t {}'.format( i, np.max(adiff), delta))
                
                
            # for manipulation of following files, the delta of the last datapoint(s) is needed
            # for reversion of the manipulation to manipulat transition to previous file 
            # the delta of the first datapoint(s) is needed
            
            # save delta of last value 
            
            
            acopy = np.copy(measurement[:,i])
            acopy = acopy[~np.isnan(acopy)]
            
            first_delta = acopy[0] -first_value
            last_delta = acopy[-1] - last_value
            
            if first_delta != 0 and last_delta!= 0 and first_delta != last_delta:
                warnings.warn('Do not know if first or last delta should be taken. Taking first'+ str(first_delta) + str(last_delta))
                deltas[i] = first_delta
            elif first_delta != 0:
                deltas[i] = first_delta
            elif last_delta != 0:
                deltas[i] = last_delta
            
    
    def manipulate_transition():
        #if plt: print(fig)
        #STEP 2
        ################################################################
        # manipulate jumps between consecutive files
        # compare manipulated head of current manipulated file with tail of previous file
        # has to account for time gaps between consecutive files, since sample 
        # lag difference may be significantly greater than true derivative
        # several causes for jumps are possible:
        #    1. jumps in this file were manipulated by delta
        #        1.1. manipulation was done to "right" side of the range where jumps occur
        #        1.2. manipulation was done to "wrong" side of the range where jumps occur
        #    2. there were no jumps in this file
        #        2.1. this file in on the "right" side of the range where jumps occur
        #        2.2. this file in on the "right" side of the range where jumps occur
        #
        # derivative based assessment of jumps fails, if too many values are missing
        # there for a least-squares fit of two linear functions with equal slope,
        # but different intercept values is done 
        # if the difference of the intercept values is greater than transition threshold
        # the file has to be manipulated        
        #time_gap = (start_time - previous_start_time - datetime.timedelta(seconds=previous_a.shape[0]/sample_rate)).total_seconds()
        #print(time_gap)
        if previous_a is not None and previous_delta is not None:
            # set head and tail to ten seconds @ sampling rate of 20 Hz
            
            tail_ind = int(previous_a.shape[0] - head_tail_length*sample_rate)
            if not tail_ind > 0:
                warnings.warn('Tail is too short. Trying to manipulate transition zone anyway')
                tail_ind = 0
            
            time_tail = np.array([previous_start_time+datetime.timedelta(seconds=i/sample_rate) for i in range(tail_ind, previous_a.shape[0])])
            this_tail =previous_a[tail_ind:,i]
            
            tail_nnan = ~np.isnan(this_tail)
            this_tail = this_tail[tail_nnan]
            time_tail = time_tail[tail_nnan]
            
            head_ind = int(head_tail_length*sample_rate)
            if not measurement.shape[0]>head_ind:
                warnings.warn('Head is too short. Trying to manipulate transition zone anyway')
                head_ind = measurement.shape[0]
            
            time_head = np.array([start_time+datetime.timedelta(seconds=i/sample_rate) for i in range(head_ind)])
            this_head = measurement[:head_ind,i]
                     
            head_nnan = ~np.isnan(this_head)
            this_head = this_head[head_nnan]
            time_head = time_head[head_nnan]
            
            transition = np.concatenate((this_tail,this_head))
            
            #trans_ind = ~np.isnan(transition)            
            #transition = transition[trans_ind]            
            
            time_transition = np.concatenate((time_tail, time_head))
            #time_transition = time_transition[trans_ind]
            
            # prepare for least squares estimation of two linear functions, 
            # with equal slope and different intercept values for tail and head
            
            diff_time = time_transition - start_time
            diff_time = np.array([dt.total_seconds() for dt in diff_time])            
            
            ones_tail = np.zeros_like(time_transition, dtype=float)
            ones_tail[:len(time_tail)] = 1
            
            ones_head = np.zeros_like(time_transition, dtype=float)
            ones_head[len(time_tail):] = 1
            
            x = np.vstack((diff_time,ones_tail,ones_head))
            lsq_x = np.linalg.pinv(x)
            
            slope,intercept_tail,intercept_head = transition.dot(lsq_x)
            diff = intercept_head-intercept_tail
              
                
            if plt2:
                plot.figure()
                plot.plot(time_transition, transition)
                
                y_1_new =  diff_time*slope + intercept_tail
                y_2_new =  diff_time*slope + intercept_head                
                plot.plot(time_transition,y_1_new, ls='solid',marker='')
                plot.plot(time_transition,y_2_new, ls='solid',marker='')              

                ax1=plot.gca()
            
            if plt or plt2:
                fig=plot.gcf()

            
            if np.abs(diff)>transition_threshold:                

                slope,intercept_tail,intercept_head = transition.dot(lsq_x)
                diff = intercept_head-intercept_tail
                
                # in the previous step the clusters are always moved towards the lower cluster (smaller mean),
                # i.e. it holds: delta <= 0
                # there are multiple jump zones in each channel, 
                # these are triggered for instance if there are sharp drops/increases in temperature
                # this results in six cases to distinguish
                #
                # 1 single jump to the top, with clustering
                #    delta < 0, diff = 0 
                #        save delta
                #        done in previous step (clustering)
                #------------------------------------------------------ diff > 0
                # 2 single jump to the top, without clustering
                #    delta = 0, diff > 0  len(previous_delta)==1
                #        add previous_delta
                #        save previous_delta
                # 3 multiple jumps to the top, with clustering
                #    delta < 0, diff > 0
                #        add previous_delta(s) (while checking diff)
                #        save previous_delta(s) and delta
                #            special care has to be taken:
                #                how many previous deltas have to be added
                #                overwrite the last previous_delta with delta or append delta
                #                
                # 4 multiple jumps to the top, without clustering
                #    delta = 0, diff > 0, len(previous_delta)>1
                #        add previous_delta(s) (while checking diff)
                #        save previous_delta(s)
                #        assumption: all previous_deltas will be used and saved
                # ------------------------------------------------------ diff <0
                # 5 single jump to the bottom, with clustering
                #    delta < 0, diff < 0
                #        invert and add delta
                #        save delta
                # 6 single jump to the bottom, without clustering
                #    delta = 0,  diff < 0, len(previous_delta)==1
                #        add previous_delta
                #        save previous_delta
                # 7 multiple jumps to the bottom, with clustering
                #    delta < 0, diff < 0
                #        invert and add delta
                #        add previous_delta(s) (while checking diff)
                #        save previous_delta(s) and delta
                #            special care has to be taken:
                #                how many previous deltas have to be added
                #                overwrite the last previous_delta with delta or append delta
                # 8 multiple jumps to the bottom, without clustering
                #    delta = 0, diff < 0, len(previous_delta)>1
                #        add previous_delta(s) (while checking diff)
                #        save previous_delta(s)
                #        assumption: all previous_deltas will be used and saved
        
                this_delta = deltas[i]    
                deltas[i] = []
                # in deltas the delta closest to the current "raw" value is first and further deltas follow
                # the delta closest to the "manipulated" value is last
                # s.t. when manipulating we apply previous deltas in reversed order 
                # and can therefore replace/delete unneeded deltas
                
                if isinstance(previous_delta[i], list):
                    previous_deltas = previous_delta[i]
                else:
                    previous_deltas = [previous_delta[i]]
                
                if diff>0:
                    
                    if this_delta ==0 and len(previous_deltas)==1 :
                                
                        if  np.abs(this_tail[-1]-this_head[0]-previous_deltas[0])<np.abs(this_tail[-1]-this_head[0]):
                                
                        #if np.abs(diff+previous_deltas[0])<np.abs(diff):
                            # 2 single jump to the top, without clustering
                            #    delta = 0, diff > 0  len(previous_delta)==1
                            #        add previous_delta
                            #        save previous_delta
                            logger.debug('{}\t transition zone to previous (single) \t {} \t{}'.format(i, diff,previous_deltas))
                            measurement[:,i] += previous_deltas[0]
                            deltas[i] = previous_deltas
                            
                            this_head = measurement[:head_ind,i]     
                            this_head = this_head[head_nnan]
                             
                            transition = np.concatenate((this_tail,this_head))       
                            #transition = transition[trans_ind]    
                            
                            slope,intercept_tail,intercept_head = transition.dot(lsq_x)
                            diff_new = intercept_head-intercept_tail
                            
                            # check for  false-positives, manipulation should always reduce diff
                            #if np.abs(diff_new)>np.abs(diff):
                            #    print(i, '\t undo', diff_new)
                            #    measurement[:,i] -= previous_deltas[0]
                            #    deltas[i] = []
                                
                        
                    elif this_delta < 0:     
                
                        # 3 multiple jumps to the top, with clustering
                        #    delta < 0, diff > 0
                        #        add previous_delta(s) (while checking diff)
                        #        save previous_delta(s) and delta
                        
                        # if this is the first file where the second jump zone occurs, this_delta has to be added deltas[i]
                        # if this is any other file, deltas[i] has to be overwritten by this_delta
                        # if adding all previous_deltas except the first yields diff<threshold:
                        #    insert this delta 
                        # else:
                        #    adding first previous_delta should now yield diff<threshold:
                        #    insert this delta     
                        
                        # we can probably just follow the procedure for case 4
                                                
                        #for this_previous_delta in previous_deltas[1:]:
                        for this_previous_delta in reversed(previous_deltas[1:]):
                            if not np.abs(this_tail[-1]-this_head[0]-this_previous_delta)<np.abs(this_tail[-1]-this_head[0]):
                                break
                            #if not np.abs(diff+this_previous_delta)<np.abs(diff):
                            #    break
                            logger.debug('{}\t transition zone to previous (multiple, with clustering) \t{} \t{} \t{}'.format( i,diff,previous_deltas,this_delta))
                        
                            measurement[:,i] += this_previous_delta
                            
                            #deltas[i].append(this_previous_delta)
                            deltas[i].insert(0,this_previous_delta)
                            
                            this_head =measurement[:head_ind,i]        
                            this_head = this_head[head_nnan]
                               
                            transition = np.concatenate((this_tail,this_head))       
                            #transition = transition[trans_ind]    
                            
                            slope,intercept_tail,intercept_head = transition.dot(lsq_x)
                            diff_new = intercept_head-intercept_tail
                            
                            # check for  false-positives, manipulation should always reduce diff
                            #if np.abs(diff_new)>np.abs(diff):
                            #    print(i, '\t undo', diff_new)
                            #    measurement[:,i] -= this_previous_delta
                            #    del deltas[i][0]                    
                            #    break                    
                            
                            diff=diff_new

                            #if np.abs(diff_new)<=transition_threshold:
                            #    break
                            
                        else:# i.e. len(previous_deltas) == 1 or previous_step did not yield 
                            try:
                                np.abs(this_tail[-1]-this_head[0]-previous_deltas[0])<np.abs(this_tail[-1]-this_head[0])
                            except:
                                logger.exception('error')
                                raise
                            if np.abs(this_tail[-1]-this_head[0]-previous_deltas[0])<np.abs(this_tail[-1]-this_head[0]):
                            #if np.abs(diff+previous_deltas[0])<np.abs(diff):
                                logger.debug('{}\t transition zone to previous (multiple, with clustering) \t{} \t{} \t{}'.format( i,diff,previous_deltas,this_delta))
                            
                                this_previous_delta = previous_deltas[0]
                                
                                deltas[i].insert(0,this_previous_delta)
                                
                                measurement[:,i] += this_previous_delta
                                
                                this_head = measurement[:head_ind,i]         
                                this_head = this_head[head_nnan]
                                  
                                transition = np.concatenate((this_tail,this_head))       
                                #transition = transition[trans_ind]    
                                
                                slope,intercept_tail,intercept_head = transition.dot(lsq_x)
                                diff_new = intercept_head-intercept_tail
                                
                                # check for  false-positives, manipulation should always reduce diff
                                #if np.abs(diff_new)>np.abs(diff):
                                #    print(i, '\t undo', diff_new)
                                #    measurement[:,i] -= this_previous_delta
                                #    del deltas[i][0]
                                    
                                
                                if not np.abs(diff_new)<=transition_threshold:
                                    warnings.warn('all previous_deltas were added, diff {} should now be smaller than {}'.format(np.max(np.abs(diff[~np.isnan(diff)])),transition_threshold))
                                
                        deltas[i].insert(0,this_delta)  
                          
                        
                    elif this_delta ==0 and len(previous_deltas)>1:

                        # 4 multiple jumps to the top, without clustering
                        #    delta = 0, diff > 0, len(previous_delta)>1
                        #        add previous_delta(s) (while checking diff)
                        #        save previous_delta(s)
                        #        assumption: all previous_deltas will be used and saved 
                        #        can this function overshoot? i.e. there are more previous_deltas than necessary
                        # if we are coming from a range with clustering and manipulation to previous, the clustering delta was saved but not applied
                        # but are now at the lower end of the previous clusters we have to apply previous_delta backwards while checking diff
                        # this way we can avoid over shooting 
                             
                        for this_previous_delta in reversed(previous_deltas):
                            if not np.abs(this_tail[-1]-this_head[0]-this_previous_delta)<np.abs(this_tail[-1]-this_head[0]):
                                break
                            #if not np.abs(diff+this_previous_delta)<np.abs(diff):
                            #    break
                            
                            logger.debug('{} \t transition zone to previous (multiple, w/o clustering) \t {} \t{}'.format(i,diff,previous_deltas))
                            measurement[:,i] += this_previous_delta                            
                            
                            deltas[i].insert(0,this_previous_delta)
                            
                            this_head =measurement[:head_ind,i]           
                            this_head = this_head[head_nnan]
                            
                            transition = np.concatenate((this_tail,this_head))    
                            #transition = transition[trans_ind]    
                            
                            slope,intercept_tail,intercept_head = transition.dot(lsq_x)
                            diff_new = intercept_head-intercept_tail
                            # check for  false-positives, manipulation should always reduce diff
                            #if np.abs(diff_new)>np.abs(diff):
                            #    print(i, '\t undo', diff_new)
                            #    measurement[:,i] -= this_previous_delta
                            #    del deltas[i][0]
                            #    break                            
                            
                            diff=diff_new
                            #if np.abs(diff_new)<=transition_threshold:
                            #    break
                    else:
                        warnings.warn('This should not have happened!\n diff {}, deltas[i] {}, previous_deltas {}'.format(diff, this_delta, previous_deltas))
                
                elif diff<0:   
                    
                    if this_delta < 0:

                        # 5 single jump to the bottom, with clustering
                        #    delta < 0, diff < 0
                        #        invert and add delta
                        #        save delta
                        
                        # 7 multiple jumps to the bottom, with clustering
                        #    delta < 0, diff < 0
                        #        invert and add delta
                        #        add previous_delta(s) (while checking diff)
                        #        save previous_delta(s) and delta
                        #            special care has to be taken:
                        #                how many previous deltas have to be added
                        #                overwrite the last previous_delta with delta or append delta
                        
                        # cases 5 and 7 have to be distinguished by trial-and-error
                        # case 5

                        if  np.abs(this_tail[-1]-this_head[0]-this_delta*-1)<np.abs(this_tail[-1]-this_head[0]):
                        #if np.abs(diff-this_delta)<np.abs(diff):
                            logger.debug('{} \t reverting transition zone \t {} \t {}'.format( i,diff, this_delta))
                            this_delta *= -1
                            measurement[:,i] += this_delta
                                
                            deltas[i].append(this_delta)
                            
                            this_head =measurement[:head_ind,i]         
                            this_head = this_head[head_nnan]
                              
                            transition = np.concatenate((this_tail,this_head))    
                            #transition = transition[trans_ind]    
                            
                            slope,intercept_tail,intercept_head = transition.dot(lsq_x)
                            diff_new = intercept_head-intercept_tail
                            
                            # check for  false-positives, manipulation should always reduce diff
                            #if np.abs(diff_new)>np.abs(diff):
                            #    print(i, '\t undo', diff_new)
                            #    measurement[:,i] -= this_delta
                            #    del deltas[i][-1]                       
                            #        
                            #    this_head =measurement[:head_ind,i]      
                            #    transition = np.concatenate((this_tail,this_head))    
                            #    transition = transition[trans_ind]    
                            #    
                            #    slope,intercept_tail,intercept_head = transition.dot(lsq_x)
                            #    diff_new = intercept_head-intercept_tail
                            diff = diff_new
                            
                            # case 7
                            if np.abs(diff)>transition_threshold:
    
                                for this_previous_delta in reversed(previous_deltas):
                                    
                                    if not np.abs(this_tail[-1]-this_head[0]-this_previous_delta)<np.abs(this_tail[-1]-this_head[0]):
                                        break
#                                     if not np.abs(diff+this_previous_delta)<np.abs(diff):
#                                         break
                                    logger.debug('{}\t transition zone to previous (multiple, with clustering) \t{}\t{}'.format(i,diff, previous_deltas))
                            
                                    measurement[:,i] += this_previous_delta
                                
                                    deltas[i].insert(1, this_delta)
                                    
                                    this_head =measurement[:head_ind,i]         
                                    this_head = this_head[head_nnan]
                              
                                    transition = np.concatenate((this_tail,this_head))    
                                    #transition = transition[trans_ind]    
                                    
                                    slope,intercept_tail,intercept_head = transition.dot(lsq_x)
                                    diff_new = intercept_head-intercept_tail
                                    
                                    # check for  false-positives, manipulation should always reduce diff
                                    #if np.abs(diff_new)>np.abs(diff):
                                    #    print(i, '\t undo', diff_new)
                                    #    measurement[:,i] -= this_previous_delta
                                    #    del deltas[i][1]
                                    #    break                    
                            
                                    diff=diff_new
                                    
                                    #if np.abs(diff_new)<=transition_threshold:
                                    #    break
                                else:
                                    if np.max(np.abs(diff[~np.isnan(diff)]))>transition_threshold:
                                        warnings.warn('all previous_deltas were added, diff {} should now be smaller than {}'.format(np.max(np.abs(diff[~np.isnan(diff)])),transition_threshold))
                    
                        
                    elif this_delta== 0 and len(previous_deltas)==1:
 
                                    
                        if np.abs(this_tail[-1]-this_head[0]-previous_deltas[0])<np.abs(this_tail[-1]-this_head[0]):
 
                        #if np.abs(diff+previous_deltas[0])<np.abs(diff):
                            # 6 single jump to the bottom, without clustering
                            #    delta = 0,  diff < 0, len(previous_delta)==1
                            #        add previous_delta
                            #        save previous_delta
                            logger.debug('{}\t transition zone to previous (single) \t{}\t{}'.format(i,diff, previous_deltas))
                            measurement[:,i] += previous_deltas[0]
                            deltas[i] = previous_deltas
    
                            this_head = measurement[:head_ind,i]           
                            this_head = this_head[head_nnan]
                            
                            transition = np.concatenate((this_tail,this_head))    
                            #transition = transition[trans_ind]    
                            
                            slope,intercept_tail,intercept_head = transition.dot(lsq_x)
                            diff_new = intercept_head-intercept_tail
                            
                            # check for  false-positives, manipulation should always reduce diff
                            #if np.abs(diff_new)>np.abs(diff):
                            #    print(i, '\t undo', diff_new)
                            #    measurement[:,i] -= previous_deltas[0]
                            #    deltas[i]=[]
                             
                        
                    elif this_delta == 0 and len(previous_deltas)>1:

                        # 8 multiple jumps to the bottom, without clustering
                        #    delta = 0, diff < 0, len(previous_delta)>1
                        #        add previous_delta(s) (while checking diff)
                        #        save previous_delta(s)
                        #        assumption: all previous_deltas will be used and saved
                        
                        # can this function overshoot? i.e. there are more previous_deltas than necessary
                        # if we are coming from a range with clustering and manipulation to previous, the clustering delta was saved but not applied
                        # but are now at the lower end of the previous clusters we have to apply previous_delta backwards while checking diff
                        # this way we can avoid over shooting 
                             
                        for this_previous_delta in reversed(previous_deltas):
                                    
                            if not np.abs(this_tail[-1]-this_head[0]-this_previous_delta)<np.abs(this_tail[-1]-this_head[0]):
                                break
#                             if not np.abs(diff+this_previous_delta)<np.abs(diff):
#                                 break
                            
                            logger.debug('%s\t transition zone to previous (multiple, w/o clustering) \t%s\t%s', i, diff, previous_deltas)
                            measurement[:,i] += this_previous_delta                            
                            
                            deltas[i].insert(0,this_previous_delta)
                            
                            this_head =measurement[:head_ind,i]          
                            this_head = this_head[head_nnan]
                             
                            transition = np.concatenate((this_tail,this_head))    
                            #transition = transition[trans_ind]    
                            
                            slope,intercept_tail,intercept_head = transition.dot(lsq_x)
                            diff_new = intercept_head-intercept_tail
                            
                            # check for  false-positives, manipulation should always reduce diff
                            #if np.abs(diff_new)>np.abs(diff):
                            #    print(i, '\t undo', diff_new)
                            #    measurement[:,i] -= previous_deltas[0]
                            #    deltas[i]=[]
                            #    break                    
                            
                            diff=diff_new
                            
                            #if np.abs(diff_new)<=transition_threshold:
                            #    break
                    else:
                        warnings.warn('This should not have happened!\n diff {}, deltas[i] {}, previous_deltas {}'.format(diff, deltas[i], previous_deltas))
            
                this_tail =previous_a[tail_ind:,i]
                this_head =measurement[:head_ind,i]     
                this_head = this_head[head_nnan]
                
                transition = np.concatenate((this_tail,this_head))
                #transition = transition[trans_ind]    
                
                if plt2:
                    ax1.plot(time_transition, transition)
                
                if not deltas[i]:
                    deltas[i] = 0   
            if plt2:
                plot.show()
                while plot.fignum_exists(fig.number):  # pylint: disable=possibly-used-before-assignment
                    plot.pause(2)

        if plt:
            axes[0].plot(measurement[:,i], alpha=.3,color='blue')
            bins=np.arange(np.nanmin(measurement[:,i])-0.0005,np.nanmax(measurement[:,i])+0.0015,0.001)    
            axes[1].hist(measurement[:,i],orientation='horizontal', alpha=.3, bins=bins,align='left',range=(bins.min(),bins.max()))
            diff = measurement[1:,i]-measurement[:-1,i]
            diff = diff[~np.isnan(diff)]
            if np.any(np.abs(diff)>transition_threshold/2):
                logger.debug('data still not consistent w.r.t. derivative')
            plot.show()
            
            
        
    def manipulate_missing_datapoints():
        # STEP 4:
        ################################################################
        # interpolate missing datapoints
        # identify contiguous missing datapoints   
        data = np.where(np.isnan(measurement[:,i]))[0]
        if len(data):
            #k: key, g: group
            # list(enumerate(data)) --> [(0, 3599), (1, 4402), (2, 4403), (3, 8916)]
            # [k for k, g in groupby('AAAABBBCCDAABBB')] --> A B C D A B
            # [k for k, g in groupby(enumerate(data), lambda i_x:i_x[0] - i_x[1]) -> [-3599, -4401, -8913]
            for k, g in groupby(enumerate(data), lambda i_x:i_x[0] - i_x[1]):
                # interpolate using 2 adjacent datapoints on each side
                # interpolate missing datapoints and fit into quantization scheme
                missing = list(map(itemgetter(1), g)) # returns the items from data for the group k
                
                len_m = len(missing)
                
                interp_x = []
                interp_y = []
                j=-1
                while True:
                    #some of the support points may be nan
                    if (missing[0]+j)<0:
                        break
                    if np.isnan(measurement[missing[0]+j,i]):
                        j -= 1
                    else:
                        interp_x.insert(0,missing[0]+j)
                        interp_y.insert(0,measurement[missing[0]+j,i])
                        j -= 1
                    if len(interp_x)>=2:
                        break
                    
                j = 1
                while True:
                    if (missing[-1]+j)>=measurement.shape[0]:
                        break
                    if np.isnan(measurement[missing[-1]+j,i]):
                        j += 1
                    else:
                        interp_x.append(missing[-1]+j)
                        interp_y.append(measurement[missing[-1]+j,i])
                        j += 1
                    if len(interp_x)>=4:
                        break 

                spl = scipy.interpolate.InterpolatedUnivariateSpline(interp_x, interp_y, k=min(3,len(interp_x)-1), ext=1)
                interp_y_new = spl(missing)
                
                # fit interpolated valued into quantization bins
                bins=np.arange(np.nanmin(measurement[:,i]),np.nanmax(measurement[:,i])+0.001,0.001)
                bins = np.reshape(bins, (1,bins.shape[0]))
                ind = np.argmin(np.abs(bins-np.expand_dims(interp_y_new,axis=1)),axis=1)#[0]
                bins = np.reshape(bins,(bins.shape[1],))
                
                interp_y_new = bins[ind]                
                measurement[missing,i]=interp_y_new    
    
    observe_i_jumps = 1000
    observe_i_transition = 1000
    
    # DO NOT CHANGE THIS is has been carefully calibrated
    transition_threshold = 0.025 # in seconds, what is considered an artificial jump and what is caused by environmental factors, i.e. sharp increases in temperature
    # lowering to 0.025 was required by file  Peaks_2016-12-21 09-36-46_1018 to 1019
    
    head_tail_length = 120# should cancel out dynamics e.g. several times the first natural period

    num_channels = measurement.shape[1]
    
    deltas=[0 for i in range(num_channels)]

    for i in range(0,num_channels):
        
                
        if i==observe_i_jumps:
            plt=True
        else:
            plt=False
                    
        if i==observe_i_transition:
            plt2=True
        else:
            plt2=False  
            
        if plt:
            fig,axes=plot.subplots(nrows=1, ncols=2, sharex=False, sharey=True,gridspec_kw={'width_ratios':[.8,.2]},tight_layout=1)#plot.figure()
            fig2,axes2=plot.subplots(nrows=1, ncols=2, sharex=False, sharey=True,gridspec_kw={'width_ratios':[.8,.2]},tight_layout=1)
            

        manipulate_jumps()

        manipulate_transition()

        if plt:
            while plot.fignum_exists(fig.number):
                plot.pause(2)
        
        manipulate_missing_datapoints()
        
    return measurement, deltas

# def save_asc(a, startTimestamp, samples, inFileName, outFileName):
#     
#     cols = a.shape[1]
#     
#     SeparatorMajor = ''
#     SeparatorMinor = ''
#     
#     sampleValue = u'{:+e}'.format(a[0][0])
#     
#     for i in range((len(sampleValue)//4+1)*4*(cols+1)):
#         SeparatorMajor += '='
#         SeparatorMinor += '-'
#     
#     fileHeader = u"""
# {}
# file created with {} which is a Labview-bin-file reader
# Bauhaus-University Weimar, Institut of Structural Mechanics, Version 0.1
# {}
# ASCII FILE of: {}
# 
# {}
# Starting time of measured data:  {:%Y-%m-%d %H:%M:%S.%f}
# Sampling rate:       {}
# Number of channels:  {}
#     """.format(SeparatorMajor,sys.argv[0],SeparatorMinor,inFileName,
#                                       SeparatorMinor,startTimestamp,samples/4,cols/2)
#     for Name in namesChannels[channel]:
#         fileHeader += u'{:13.{}}\t'.format(Name,len(sampleValue))
#         fileHeader += u'{:13.{}}\t'.format(Name,len(sampleValue))
#     fileHeader += u'\n'
#     for Unit in ['Wl[nm]','WP[]']*int(cols/2):    
#         fileHeader += u'{:13.{}}\t'.format(Unit,len(sampleValue))
#     fileHeader += u'\n'
#     fileHeader += u'Datasize: [{},{}]\n'.format(rows,cols)
#     fileHeader += SeparatorMajor
#     fileHeader += u'\n'
#     fileHeader += (u"Minimum|Maximum|Mean|StandardDeviation|Energy|MissingValues\n")
#     fileHeader += (u''.join(u'%+10.6e\t' % d for d in (np.nanmin(a,axis=0)).tolist()))
#     fileHeader += (u'\n')
#     fileHeader += (u''.join(u'%+10.6e\t' % d for d in (np.nanmax(a,axis=0)).tolist()))
#     fileHeader += (u'\n')
#     fileHeader += (u''.join(u'%+10.6e\t' % d for d in (np.nanmean(a, dtype=np.float64,axis=0)).tolist()))
#     fileHeader += (u'\n')
#     fileHeader += (u''.join(u'%+10.6e\t' % d for d in (np.nanstd(a, dtype=np.float64,axis=0)).tolist()))
#     fileHeader += (u'\n')
#     fileHeader += (u''.join(u'%+10.6e\t' % d for d in (np.nansum(a*a, dtype=np.float64,axis=0)/(samples/4)).tolist()))
#     fileHeader += (u'\n')
#     fileHeader += (u''.join(u'%+10.6e\t' % d for d in (np.sum(np.isnan(a), axis=0).tolist())))
#     fileHeader += (u'\n')
#     fileHeader += (u"%s\n%s\n%s\n%s\n" % (SeparatorMajor,SeparatorMajor,SeparatorMajor,u"TIME HISTORY"))
#     if save:
#         np.savetxt(outFileName, a, header = fileHeader, fmt = '%10.6e', delimiter = '\t')
#     print(startTimestamp,'\t',endTimestamp,'\t',rows,'\t',rows/samples*4)
#     
#     # if the internal clock of the measurement system was changed e.g. due to automatic time synchronization, 
#     # the endTimestamp is wrong
#     # this happened at 2017-01-01 00:55:24.871166 (File: Peaks_2016-12-24 22-52-23_2037)
#     # however manipulating all following startTimeStamps would be to error-prone
#     # therefore cut off all exceeding values
#     endTimestamp_calc = startTimestamp+datetime.timedelta(seconds=rows/samples*4)
#     if int((endTimestamp_calc-endTimestamp).total_seconds())>0:
#         cut_off=int(-np.ceil((endTimestamp_calc-endTimestamp).total_seconds()*samples/4))
#         print('Calculated time is {} s ahead of endTimestamp. Cutting off {} datapoints.'.format((endTimestamp_calc-endTimestamp).total_seconds(),-cut_off))
#         a = a[:cut_off,:]
#         
#     # i.e. derivative can not be computed properly, interpolation fails, etc.
#     # removing nans
# #     while True:
# #         if np.any(np.isnan(a[:,0])):
# #             a = a[:,1:]
# #         else:
# #             break
# #     while True:
# #         if np.any(np.isnan(a[:,-1])):
# #             a = a[:,:-1]
# #         else:
# #             break       
#         
#     return headers, ['nm' for header in headers], startTimestamp, samples/4, a
#     
# def main():
#     import os
#     # find files that have not been converted
#     #toConvert = set(file.strip('.bin') for file in os.listdir(filePath)).difference(set(file.strip('.asc') for file in os.listdir(outFilePath)))
#     toConvert = set(file.strip('.bin') for file in os.listdir(filePath))
#     
#     #metadata shelve
#     import shelve
#     meta_shelve = shelve.open('meta_shelve_2.slv', 'c')
#     
#     arrays=[None]
#     deltas = [None]
#     start_times = [None]
#     plt=1
#     save=0
#     raw=1
#     start_at=28590#8142,8142
#     #6@7750
#     #30@8125
#     # alles +6
#     #48@2650,
#     #30@2675,
#     #4875,
#     #30@5525,
#     #30@5550,
#     #30@5575,
#     #30,48@5625,
#     #30@5650ff,...
#     #6000, 
#     #30@8100ff, 
#     #30@11925,
#     #46@12075,
#     #46@12175, 
#     #30@17075, 
#     #17733
#     #start_at=1903
#     
#     if plt:
#         plot.ion()
#         fig,axes = plot.subplots(nrows=3, ncols=9, sharex=True,sharey=False)
#         for row in range(3):
#             for col in range(9):
#                 axes[row,col].plot([],[],label=str((row*9+col)*2))
#                 axes[row,col].legend()
#         if raw:
#             fig2,axes2 = plot.subplots(nrows=3, ncols=9, sharex=True,sharey=True)
#             for row in range(3):
#                 for col in range(9):
#                     axes2[row,col].plot([],[],label=str((row*9+col)*2))
#                     axes2[row,col].legend()
#                 
#                         
#         fig.canvas.draw()
#         fig.canvas.flush_events()
#     # from file 478 we have some sharp (physical) drops in wavelengths, for calibration of transition_threshold
#     # from file 1540 we have multiple jumps upwards, for testing multiple jumps manipulation algorithm
#     # from file 1875 we have multiple jumps downwards,
#     delta=None
#     for i,file in enumerate(sorted(toConvert)[:]):
#         
#         if i==start_at-1 and not raw:
#             #array,start_time, sample_rate = read_bin_save_asc(file)
#             meta_dict = meta_shelve.get(file, None)
#             #meta_dict = None
#             if meta_dict:
#                 #assert start_time == meta_dict['start_time']
#                 #assert sample_rate == meta_dict['sample_rate']
#                 
#                 array = np.load(processedFilePath + file.strip('.bin') + '.npz')['array']
#                 arrays.append(array)
#                 deltas.append(meta_dict['delta'])
#                 start_times.append(meta_dict['start_time'])
#             else:
#                 
#                 arrays.append(None)
#                 deltas.append(None)
#                 start_times.append(None)
#             continue
#         elif i<start_at-1 and not raw: 
#             arrays.append(None)
#             deltas.append(None)
#             start_times.append(None)
#             continue
#         elif i< start_at and raw:
#             continue
#         print('File {} of {}...'.format(i,len(toConvert)))
#         #array,start_time, sample_rate = read_bin_save_asc(file)
#         headers, units, start_time, sample_rate, array = read_bin_save_asc(file)
#         if not raw:
#             array, delta = manipulated_data(array,start_time, sample_rate, arrays[i], deltas[i], start_times[i])
#             # what should be saved:
#             # array, delta, start_time, sample_rate
#             meta_dict = {'delta':delta,'start_time':start_time,'sample_rate':sample_rate}
#             if save: meta_shelve[file]=meta_dict
#             #meta_dict['array']=array
#             if save: np.savez_compressed(processedFilePath + file.strip('.bin') + '.npz', array=array,meta_dict=meta_dict)
#         
#         
#         
#         time=[start_time+datetime.timedelta(seconds=i/sample_rate) for i in range(array.shape[0])]
#         print(time[0], time[-1], array.shape, sample_rate)
#         if plt:
#             for row in range(3):
#                 for col in range(9):
#                     axes[row,col].plot(time,array[:,(row*9+col)*2], color='blue',alpha=.3,marker=',',ls='none',markersize=2)
#                     axes[row,col].relim()
#                     axes[row,col].autoscale_view()
#             if not save:
#                 fig.canvas.draw()
#                 fig.canvas.flush_events()
#             if raw:
#                 for row in range(3):
#                     for col in range(9):
#                         axes2[row,col].plot(time,array[:,(row*9+col)*2+1], color='blue',alpha=.3,marker=',',ls='none',markersize=2)
#                         axes2[row,col].relim()
#                         axes2[row,col].autoscale_view()
#                 if not save:
#                     fig2.canvas.draw()
#                     fig2.canvas.flush_events()
#                 
#             plot.pause(10)
#             
#         arrays.append(array)
#         if not raw:
#             deltas.append(delta)
#             start_times.append(start_time)
#         if plt and save and i>start_at and not i%25:
#             fig.canvas.draw()
#             fig.canvas.flush_events()
#             if raw: plot.savefig('{}_{}_raw.png'.format(i-25,i))
#             else: plot.savefig('{}_{}.png'.format(i-25,i))
#             for row in range(3):
#                 for col in range(9):
#                     axes[row,col].clear()
#             for row in range(3):
#                 for col in range(9):
#                     axes[row,col].plot(time,array[:,(row*9+col)*2], color='blue',alpha=.3,marker=',',ls='none',markersize=2,label=str((row*9+col)*2))
#                     axes[row,col].relim()
#                     axes[row,col].autoscale_view()
#                     axes[row,col].legend()
#                     
#             fig.canvas.draw()
#             fig.canvas.flush_events()
#             
#             
#                 
#     #array=np.vstack(arrays)
# 
#     plot.pause(600)
        #print(array.shape)
# def plot_deltas():
#     import shelve
#     meta_shelve = shelve.open('meta_shelve_2.slv', 'c')
# 
#     
#     
#     deltas = [[] for i in range(27)]
#     start_times = [[] for i in range(27)]
#     for meta_dict in meta_shelve.values():
#         for channel in range(27):
#             try:
#                 for delta in meta_dict['delta'][channel]:
#                     deltas[channel].append(delta)
#                     start_times[channel].append(meta_dict['start_time'])
#             except TypeError:
#                 deltas[channel].append(meta_dict['delta'][channel])                
#                 start_times[channel].append(meta_dict['start_time'])
#         
#     #deltas = np.array(deltas)
#     #plot.ion()
#     #fig,axes = plot.subplots(nrows=3, ncols=9, sharex=True,sharey=False)
#     for row in range(3):
#         for col in range(9):
#             plot.figure()
#             plot.plot(start_times[row*col],deltas[row*col],label=str((row*9+col)*2),ls='none',marker=',', markersize=1)
#             plot.legend()   
#     plot.show() 

def read_strain_txt(path):
    
    comb_df = []
    comb_headers= []
    comb_units = []
    for j in [2,3,4]:
        if not (os.path.exists(path+'strain-{}.txt'.format(j)) or os.path.exists(path+'strain-{}.txt.bz2'.format(j))) :
            warnings.warn('File missing: {}strain_{}.txt'.format(path,j))
            return None

    for j in [2,3,4]:
        if os.path.exists(path+'strain-{}.txt.bz2'.format(j)):
            this_path = path+'strain-{}.txt.bz2'.format(j)
            f  = bz2.open(this_path, 'rt', encoding='cp1252')
            try:
                f.seek(0,2)
                f.seek(0,0)
            except EOFError:
                warnings.warn('BZ2File is corrupted: {}'.format(this_path))
                return None
        else:
            
            this_path = path+'strain-{}.txt.bz2'.format(j)
            f= open(this_path, encoding='cp1252')
            
        logger.debug(this_path)
        # iterate over first section of file (information about acquisition system and settings)
        # continue until empty line is encountered
        sampling_rate = None
        line = f.readline()
        while line.strip():
            if 'Sample Speed' in line:
                sampling_rate = float(line.split(':')[-1])
            line=f.readline()
        curr_pos = f.tell()
                
        # reads second section of file (information about sensors/FBGS and settings/engineered quantities
        # read settings_header
        settings_header = f.readline().rstrip('\r\n').rstrip('\t ').split('\t')
        # determine number of rows / end of settings section, and set file pointer back to end of first section
        for nrows,line in enumerate(f): 
            if not line.strip(): 
                f.seek(curr_pos)
                break
        else:
            return None
        # read settings section
        settings = pd.read_csv(f,
                               sep='\t',
                                 header=0,
                                 names = settings_header, 
                                 nrows = nrows, 
                                 usecols=list(range(len(settings_header))))
        num_channels = len(settings)
        channel_headers = list(settings.Name)
        channel_headers = [head.strip(' ') for head in channel_headers]
        units= list(settings.Unit)

        #set cursor to end of first section and iterate over settings again
        f.seek(curr_pos)
        line=f.readline()
        i=0
        while i<100:
            if line.startswith('Date'):
                this_headers = line.strip('\r\n\t').split('\t')[:4+num_channels]
                break
            line=f.readline()
            i+=1
        else:
            raise RuntimeError('Could not set cursor to start of actual data in file.')
        
        # read csv tables using pandas fast c-engine
        dtypes={'Date':np.datetime64,
                'LineNumber':np.int32}
        
        for this_head in this_headers[4:]:
            dtypes[this_head]=np.float64
        try:
            df=pd.read_csv(f,
                        sep='\t',
                        names=this_headers,
                        parse_dates=[0], 
                        converters={3: lambda s: '1' not in s}, 
                        index_col=False,
                        decimal=',', 
                        usecols=range(4+num_channels), 
                        error_bad_lines=False,
                        warn_bad_lines=True,)
        # if malformed csv files are encountered, avoid exiting programm, but rather print the error message and return nothing
        except pandas.errors.ParserError as e:
            warnings.warn('pandas.errors.ParserError: '+str(e)+this_path)
            warnings.warn('Trying to switch from c engine to python engine.')
            f.seek(curr_pos)
            line=f.readline()
            i=0
            while i<100:
                if line.startswith('Date'):
                    this_headers = line.strip('\r\n\t').split('\t')[:4+num_channels]
                    break
                line=f.readline()
                i+=1
            else:
                raise RuntimeError('Could not set cursor to start of actual data in file.')
            df=pd.read_csv(f,
                        sep='\t',
                        names=this_headers,
                        parse_dates=[0], 
                        converters={3: lambda s: '1' not in s}, 
                        index_col=False,
                        decimal=',', 
                        usecols=range(4+num_channels), 
                        error_bad_lines=False,
                        warn_bad_lines=True,
                        engine='python')
        except Exception as e:
            warnings.warn('pandas.errors.ParserError: '+str(e)+this_path)
            return None
        
        # Rename columns, to channel_headers, read from settings section
        for column in df.columns:
            df.rename(columns={column:column.replace('TC-Probe-','').replace('SG-01-','')}, inplace=True)
        
        # merge "Date" and "Time" columns into single timestamp column
        if not df.empty:
            ms = (np.array(df.index)/50*1000000000).astype('timedelta64[ns]')+pd.Timedelta(df['Time'][0])
            ms[df['Date']>df['Date'][0]] -= pd.Timedelta('1 day')
            df['Date'] += ms
        df.drop(labels=['Time','LineNumber'],axis=1, inplace=True)
        
        comb_df.append(df)
        comb_headers.append(channel_headers)
        comb_units.append(units)
        
        f.close()
    for df in comb_df[1:]:
        comb_df[0]['System status'] | df['System status']
        df.drop(labels=['Date','System status'],axis=1, inplace=True)
        pass
    df = pd.concat(comb_df, axis=1)
    
    if df.empty:
        warnings.warn('Empty file: '+path)
        return None

    headers=sum(comb_headers,[])
    units= sum(comb_units,[]) 
    start_time=df['Date'][0] 
    sample_rate=sampling_rate 
    measurement=df.iloc[:,2:].values
    
    start_time=start_time.to_pydatetime()
    
    np.savez(path+'.npz',headers=headers, units=units, startTimestamp=start_time, sample_rate=sample_rate, measurement=measurement)
    
    return headers, units, localize(start_time), sample_rate, measurement 

        
def read_spec(file):
    data = pd.read_table(file, skiprows=40, decimal=',')
    mat = data.iloc[:,3:515].as_matrix()
    pow = mat[slice(0,None,2),:]
    wl = mat[slice(1,None,2),:]
    wl1 = wl[0,:]
    pow1 = pow[0,:]
    plot.plot(wl1,pow1)
    plot.scatter(wl1[pow1>4.9],pow1[pow1>4.9], color='red', marker='x', alpha=.5)
    plot.xlim((wl1.min(),wl1.max()))
    plot.ylim((0,100))
    plot.xlabel('Wellenlänge [nm]')
    plot.ylabel('Lichtintensität [%]')
    plot.show()
              
    
if __name__ == '__main__':

    logger.setLevel(logging.DEBUG)
    #read_spec('/vegas/scratch/womo1998/txt-3-spec.txt')

    a=read_bin('/home/towermonitoring/strain_data/binary_files_unusable/Peaks_2017-05-03 15-27-04_00001.bin', stats_only=True)
    b=read_bin('/home/towermonitoring/strain_data/binary_files_unusable/Peaks_2017-05-03 15-27-04_00002.bin', stats_only=True)
    c=read_bin('/home/towermonitoring/strain_data/binary_files_unusable/Peaks_2017-05-03 15-27-04_00003.bin', stats_only=True)
    for i in [a,b,c]: print(localize(i[2]).astimezone(pytz.UTC))
    
    #a=read_bin('/vegas/scratch/womo1998/towerdata/10-minutes/strain_data_bin/Peaks_2017-10-20 15-22-23.bin')
    #print(a)
    #read_strain_txt('/vegas/scratch/womo1998/towerdata/30-minutes/strain_data_bin_srv-grk/2019-03-12_01-00_')
    #main()
    #plot_deltas()
    # TODO: upsample data after manipulation to account for the quarter-samplerate time-lag between acquisition channels
    #main()
