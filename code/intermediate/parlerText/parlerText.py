#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
from bs4 import BeautifulSoup
import pathlib
import re
import datetime
import lib.constants as c

def parse(soup, section, bit):
    return [i.get_text() for i in soup.find_all(section, {'class': bit})]

def extractInfo(link):
    l = link.find('img')
    if l is None:
        return False
    else:
        return link.find('img')['alt'] == 'Post Comments'

def getData(filepath, outpath = c.partsPath):
    fname = pathlib.Path(filepath)
    createTime = datetime.datetime.fromtimestamp(fname.stat().st_ctime)
    with open(filepath, 'r') as f:
        dat = f.read()
    soup = BeautifulSoup(dat)
    text = parse(soup, 'div', 'card--body')
    author = parse(soup, 'span', 'author--username')
    timestamp = parse(soup, 'span', 'post--timestamp')
    interaction = soup.find(extractInfo).find_all('span',{'class': 'pa--item--count'})
    comments = interaction[0].get_text()
    echoes = interaction[1].get_text()
    upvotes = interaction[2].get_text()
    # unclear why there are multiple bits of text/author per page here, but let's just
    # select the first because that seems to be the important one?
    fileID = filepath.split('/')[-1]
    res = pd.DataFrame(index =  [fileID], data = {'text': text[0], 'author': author[0], 
                        'timestamp': timestamp[0], 'createTime': createTime,
                       'comments': comments, 'echoes': echoes, 'upvotes': upvotes})
    res.to_csv(outpath  + fileID)



def assembleCSVs(partsPath = c.partsPath, 
                 wholePath = c.wholePath):
    files = os.listdir(partsPath)
    nSuccess = len(files)
    wholeDF = pd.DataFrame(index = range(nSuccess))
    df1 = pd.read_csv(partsPath + files[0])
    for col in df1.columns:
        wholeDF[col] = ''
        
    for i in range(nSuccess):
        try:
            wholeDF = fillDF(wholeDF, partsPath + '/' + files[i], i)
        except:
            print(files[i])
        
    wholeDF.to_csv(wholePath, index = False)

# takes all the individually saved csvs and puts them into one big csv
def fillDF(wholeDF, filepath, i):
    partDF = pd.read_csv(filepath)
    wholeDF.iloc[i,:] = partDF.T.iloc[:,0]
    return wholeDF

# 
def assembleData(filepath, handlingpath = c.assemblyHandlingPath):
    files = os.listdir(filepath)
    
    nSuccess = 0
    success = [''] * len(files)
    nErrors = 0
    errors = [''] * len(files)
    
    for file in files:
        
        if (nErrors + nSuccess) % 1e4 == 0:
            print('Completing iteration {} at {}'.format(str(nErrors + nSuccess),
                                                         datetime.datetime.now())) 
            pd.Series(errors).to_csv(handlingpath + 'errors.csv', index = False)
            pd.Series(success).to_csv(handlingpath + 'success.csv', index = False)
        iterFile = filepath + '/' + file
        try: 
            getData(iterFile)
            success[nSuccess] = iterFile
            nSuccess += 1
        except:
            errors[nErrors] = iterFile
            nErrors += 1
    print('Total number of files: {}'.format(str(nErrors + nSuccess)))
    print('Total number of errors: {}'.format(str(nErrors)))
    print('Total number of success: {}'.format(str(nSuccess)))
    assembleCSVs()


if __name__ == '__main__':
    assembleData(c.partialTextPath)




