#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import os
import pathlib
import datetime
import sys
sys.path.append('/home/beeb/Insync/sei2112@columbia.edu/Google Drive/Columbia SusDev PhD/Research/My work/parler/code/lib')
import constants as c
import matplotlib.pyplot as plt
os.chdir(c.wd)


# ## In this file
# For the RDD strategy, we're interested in finding individuals who joined Parler within some window of a cutoff date (specifically, one of the dates when there was a large influx of users into the platform). This code finds, for one of these cutoff dates, the specific group of indivduals which joined shortly before/after that date.

# filters out only the individuals who fall within the cutoff period
def takeSample(dat, cutoff, lowerCutoff, upperCutoff):
    dat['post'] = dat.joined > cutoff
    dat['inSample'] = (dat.joined < upperCutoff)
    datIS = dat[dat.inSample]
    return datIS

# changes dates and window into appropriate formats
def getCutoffs(breakYear, breakMonth, breakDay, nDaysBefore, nDaysAfter):
    cutoff = datetime.datetime(year = breakYear, month = breakMonth, day = breakDay)
    lowerCutoff = cutoff - datetime.timedelta(days = nDaysBefore)
    upperCutoff = cutoff + datetime.timedelta(days = nDaysAfter)
    return cutoff, lowerCutoff, upperCutoff

# runs code over all files
def runCode(breakYear, breakMonth, breakDay, nDaysBefore, nDaysAfter):
    cutoff, lowerCutoff, upperCutoff = getCutoffs(breakYear, breakMonth, breakDay, nDaysBefore, nDaysAfter)
    final = pd.DataFrame()
    for i in range(9):
        print('Performing iteration {i} at {j}'.format(i = i, 
                                            j = str(datetime.datetime.now())))
        dat = pd.read_csv(c.userOutpath.format(i),
                  parse_dates = ['joined', 'lastseents'])
        datIter = takeSample(dat, cutoff, lowerCutoff, upperCutoff)
        final = pd.concat([final, datIter])
    final.to_csv(c.samplePath.format(str(cutoff).split(' ')[0]), 
                 index = False)
    return final

if __name__ == '__main__':
    for j in range(c.dates.shape[0]):
        df2 = runCode(c.dates.loc[j,['years']],
           c.dates.loc[j,['months']],
           c.dates.loc[j,['days']],
           c.nDaysBefore, c.nDaysAfter)
