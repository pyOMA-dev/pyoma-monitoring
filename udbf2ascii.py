#!/usr/bin/python
"""Reader for Universal Data Binary Format (UDBF) files from Gantner Instruments.

Parses the binary `.dat` files written by the Gantner Q.station controller
and returns channel time-series data as numpy arrays with associated header
metadata (channel names, units, sample rate, start timestamp).

Version 0.1 — Bauhaus-Universität Weimar, Institute of Structural Mechanics.
"""
import matplotlib.pyplot as plot

import datetime
import pytz
import os
import sys
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import struct
import io
import numpy as np
import pandas as pd
import time
import warnings


def getDataTypeChar( dActTimeDataType ):
    ######################################################################################
    # Begin Function definitions
    ######################################################################################
    #~ 0    No
    #~ 1    Boolean
    #~ 2    SignedInt8
    #~ 3    UnSignedInt8
    #~ 4    SignedInt16
    #~ 5    UnSignedInt16
    #~ 6    SignedInt32
    #~ 7    UnSignedInt32
    #~ 8    Foat
    #~ 9    BitSet8
    #~ 10    BitSet16
    #~ 11    BitSet32
    #~ 12    Double
    #~ 13    SignedInt64
    #~ 14    UnSignedInt64
    #~ 15    BitSet64    
        
    #~ mtype -- a string indicating the binary type to write.
    #~ The default is the type of data. If necessary a cast is made.
    #~ unsigned byte  : 'B', 'uchar', 'byte' 'unsigned char', 'int8',
    #~ 'integer*1'
    #~ character      : 'S1', 'char', 'char*1'
    #~ signed char    : 'b', 'schar', 'signed char'
    #~ short          : 'h', 'short', 'int16', 'integer*2'
    #~ unsigned short : 'H', 'ushort','uint16','unsigned short'
    #~ int            : 'i', 'int'
    #~ unsigned int   : 'I', 'uint32','uint','unsigned int'
    #~ int32           : 'u4', 'int32', 'integer*4'
    #~ float          : 'f', 'float', 'float32', 'real*4'
    #~ double         : 'd', 'double', 'float64', 'real*8'
    #~ complex float  : 'F', 'complex float', 'complex*8', 'complex64'
    #~ complex double : 'D', 'complex', 'complex double', 'complex*16',
    #~ 'complex128'
    getDataTypeCharDict={ 
                        2:    'b',
                        3:    'B',
                        4:    'h',
                        5:    'H',
                        6:    'i',
                        7:    'I',
                        8:    'f',
                        #~ 9:    'B',
                        12:    'd',
                        14:     'Q'
                    }
    getDataTypeSizeDict={ 
                        2:    1,
                        3:    1,
                        4:    2,
                        5:    2,
                        6:    4,
                        7:    4,
                        8:    4,
                        #~ 9:    'B',
                        12:    8,
                        14:     8
                    }

    thisDataTypeChar=getDataTypeCharDict.get(dActTimeDataType)
    thisDataTypeSize=getDataTypeSizeDict.get(dActTimeDataType)
    if thisDataTypeChar is None:
        #print dActTimeDataType, thisDataTypeChar, thisDataTypeSize
        sys.exit("Unknown data type request -- ing with ERROR")
    else:
        return (thisDataTypeChar,thisDataTypeSize)

def read_csv(inFileName, path):
    if isinstance(inFileName, str):
        fd = open( inFileName , 'rb')
    else:
        fd=inFileName
        
    headers = fd.readline().decode('cp1252').strip('\r\n').split('\t')
    # pos = fd.tell()
    units = fd.readline().decode('cp1252').strip('\r\n').split('\t')
    # fd.seek(pos)
    
    date, time, msecs=path.split('/')[-1].rstrip('.bz2').rstrip('.csv').split('_')[-3:]
    year,month, day = date.split('-')
    hour, minute, secs = time.split('-')
    
    start_time = datetime.datetime(int(year),int(month),int(day),int(hour),int(minute),int(secs),int(msecs)) 
    
    if 'Temp' in path:
        sample_rate = 20
    elif 'Wind' in path:
        sample_rate = 100
    else:
        warnings.warn('Could not determine quantity from filename. Sample_rate 0 will be returned.')
        sample_rate = 0
        
    measurement = pd.read_csv(fd,sep='\t', header=None, decimal=',').values     
    
    return headers, units, pytz.utc.localize(start_time), sample_rate, measurement

def read_bin(InFileName):
    if isinstance(InFileName, str):
        fd = open( InFileName , 'rb')
    else:
        fd=InFileName
    
    fd.seek(0, 2)
    # storePosition
    FileEnd=fd.tell()
    #raise RuntimeError('File is empty.')
    fd.seek(0, 0)
    
    isBigEndian=struct.unpack("B",fd.read(1))[0]
    
    if(isBigEndian): BigEndianFlag='>'
    else :     BigEndianFlag='<'

    
    # read VERSION
    
    version=struct.unpack(BigEndianFlag+"h",fd.read(2))
    
    version=version[0]
    # read VENDOR
    if( version > 105):
        typeVendorLen=struct.unpack(BigEndianFlag+"h",fd.read(2))[0]
        typeVendor=fd.read(typeVendorLen)
        typeVendorStr=typeVendor[:-1].decode()
    
    # read CheckSum
    if( version > 100):
        withCheckSum=struct.unpack(BigEndianFlag+"B",fd.read(1))[0]
    #print withCheckSum
    
    # read ModuleAdditionalDataLen
    ModuleAdditionalDataLen=struct.unpack(BigEndianFlag+"h",fd.read(2))[0]
    if ModuleAdditionalDataLen:
        #warnings.warn("ModuleAdditionalData has to be implemented")
        fd.read(ModuleAdditionalDataLen)
    
    # read StartTimeToDayFactor
    StartTimeToDayFactor=struct.unpack(BigEndianFlag+"d",fd.read(8))[0]

    # read dActTimeDataType
    dActTimeDataType=struct.unpack(BigEndianFlag+"h",fd.read(2))[0]

    # read aActTimeToSecondFactor
    aActTimeToSecondFactor=struct.unpack(BigEndianFlag+"d",fd.read(8))[0]

    # read StartTime
    StartTime=struct.unpack(BigEndianFlag+"d",fd.read(8))[0]

    # read SampleRate
    SampleRate=float(struct.unpack(BigEndianFlag+"d",fd.read(8))[0])
    
    # read VariableCount
    VariableCount=int(struct.unpack(BigEndianFlag+"h",fd.read(2))[0])

    # read Variable settings
    Names=['Time']
    DataDirections=[]
    DataTypes=[]
    FieldLens=[]
    Precisions=[]
    Units=['s']
    #import pprint
    for i in range(VariableCount) :
        # read Name
        NameStr=u""
        NameLen=struct.unpack(BigEndianFlag+"h",fd.read(2))[0];
        Name=fd.read(NameLen)
        NameStr=Name[:-1].decode()
        #NameStr=Name.decode('cp1252').encode('utf-8')[:-1]
        #for i in Name.decode('cp1252').encode('utf-8')[:-1]:
        #    NameStr+=i;
        #print NameStr
        
        Names.append(NameStr)
    
        # read DataDirection
        DataDirection=struct.unpack(BigEndianFlag+"h",fd.read(2))[0]
        if( DataDirection != 0):
            sys.exit("DataDirection not supported: Only output variables implemented yet!!")
        #print DataDirection
        DataDirections.append(DataDirection)
    
        # read DataType
        if( version < 102):
            sys.exit("DataType: NotSupported version of datafile!!")
        DataType=struct.unpack(BigEndianFlag+"h",fd.read(2))[0]
        #print DataType
        if DataType!=8 : logger.exception( """!!! WARNING !!! !!! WARNING !!! !!! WARNING !!! !!! WARNING !!! !!! WARNING !!!
    !!! tested only DataType 8 (found DataType: %u)
    !!! WARNING !!! !!! WARNING !!! !!! WARNING !!! !!! WARNING !!! !!! WARNING !!! 
    """ % DataType)
        DataTypes.append(DataType)
    
        # read FieldLen
        FieldLen=struct.unpack(BigEndianFlag+"h",fd.read(2))[0]
        FieldLens.append(FieldLen)
        
        # read Precision
        Precision=struct.unpack(BigEndianFlag+"h",fd.read(2))[0]
        Precisions.append(Precision)
    
        # read Unit
        if( version < 106):
            sys.exit("Unit: NotSupported version of datafile!!")
        UnitLen=struct.unpack(BigEndianFlag+"h",fd.read(2))[0];
        Unit=fd.read(UnitLen)

        #UnitStr=Unit.decode('cp1252').encode('utf-8')[:-1]
        UnitStr=Unit.decode('cp1252')[:-1]
        #UnitStr=Unit.decode()
        Units.append(UnitStr)
        
        # read AdditionalDataLen
        AdditionalDataLen=struct.unpack(BigEndianFlag+"h",fd.read(2))[0]
        if AdditionalDataLen:
            #warnings.warn("optional data for: AdditionalDataLen has to be implemented")
            fd.read(AdditionalDataLen)
    
    #check if it's a star
    while 1:
        
        currentFilePosition = fd.tell();
        char = struct.unpack(BigEndianFlag+"c",fd.read(1))[0];
        try:
            char=char.decode()
        except:
            fd.seek(currentFilePosition, 0);
            break
        if not char == '*':
            fd.seek(currentFilePosition, 0);
            break


    
    TimeDataType=getDataTypeChar(dActTimeDataType);
    
    DataTypeChars=[]
    DataTypeSize=[]
    for i in range(VariableCount):
        DataTypeChars.append(getDataTypeChar(DataTypes[i])[0]);
        DataTypeSize.append(getDataTypeChar(DataTypes[i])[1]);

    
    ######## Starttime #################
    dtReferenceDate = datetime.datetime(1900,1,1,0,0,0,0)+datetime.timedelta(days=StartTime*StartTimeToDayFactor-2)
    
    start=fd.tell()
    #dataValues=[]
    numLines = int(np.ceil(FileEnd-fd.tell())/(sum(DataTypeSize)+TimeDataType[1]))
    
    timeStamp=struct.unpack(BigEndianFlag+TimeDataType[0],fd.read(TimeDataType[1]))[0]
    startTimeStamp=timeStamp
    # logger.debug(timeStamp)
    tmp=float(timeStamp)*aActTimeToSecondFactor
    dtDeltaTime=datetime.timedelta(seconds=tmp)

    dtStartDate=dtReferenceDate+dtDeltaTime

    ReadData=[]
    thisIter=0
    startOLE=0.0
    
    dataValues=np.zeros((numLines,VariableCount+1))
    fd.seek(start,0)
    
    
    
    data = struct.unpack(BigEndianFlag+(TimeDataType[0]+''.join(DataTypeChars))*numLines,fd.read((TimeDataType[1]+sum(DataTypeSize))*numLines))
    dataValuesFlat = dataValues.ravel()
    dataValuesFlat[:] = data
    dataValues[:,0]*=aActTimeToSecondFactor

    fd.close()

    return Names, Units, pytz.utc.localize(dtStartDate), SampleRate, dataValues   #dataValues, SampleRate, Names, Units, dtStartDate
    

def main():
    path='/home/towermonitoring/towerdata/Wind_kontinuierlich__9_2017-05-03_00-00-00_000000.dat.bz2'
    #path='/vegas/scratch/womo1998/towerdata/towerdata_bin/Wind_kontinuierlich__9_2017-03-07_23-00-00_000000.dat.bz2'
    #path ='/vegas/scratch/womo1998/towerdata/towerdata_bin/Temp_konti__0_2017-01-01_00-00-00_000000.dat.bz2'
    
    import bz2
    import tzlocal
    reader_tz = tzlocal.get_localzone()
    this_time = datetime.datetime.fromtimestamp(os.path.getmtime(path), tz=reader_tz)
    # file creation time is the time, the last bit was written to the file, i.e. the end_time_stamp
    # applies for both gantner and labview systems
    # this is in UTC synchronized to local time 
    
    print(path)
    print(this_time.astimezone(pytz.UTC))
    
    zipfile  = bz2.BZ2File(path)

    abc=read_bin(zipfile)
    for a in abc:
        print(a)


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    main()
