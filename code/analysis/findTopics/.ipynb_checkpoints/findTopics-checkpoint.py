import pandas as pd
import string
import numpy as np
import os
import pathlib
import re
from datetime import datetime
import pytz
from nltk.probability import FreqDist
from scipy import stats
from wordcloud import WordCloud
from gensim import corpora, models, similarities
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
from nltk.corpus import stopwords
import sys
sys.path.append('/home/beeb/Insync/sei2112@columbia.edu/Google Drive/Columbia SusDev PhD/Research/My work/parler/code/lib')
import constants as c
import matplotlib.pyplot as plt
os.chdir(c.wd)
from statsmodels.stats.proportion import proportions_ztest
from nltk.util import ngrams
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import random

random.seed(c.SEED)

def dataIn(path, termsDict, keys, cutoffDate, outputPathDate):
    # once working with full sample:
    dat = pd.DataFrame()
    plotDat = pd.DataFrame()
    problemPosts = pd.DataFrame()
    # iterate through all files, see which ones have mentions of at least
    # one of our topics of interest and keep only those
    errors = []
    for i in range(0, (c.NPOSTS + 1)):
        if i % 10 == 0:
            dat.to_csv(os.path.join(outputPathDate, 'dat.csv'))
            plotDat.to_csv(os.path.join(outputPathDate, 'plotDat.csv'))
            problemPosts.to_csv(os.path.join(outputPathDate, 'problemPosts.csv'))
            pd.Series(errors).to_csv(os.path.join(outputPathDate, 'errors.csv'))    
        try:
            print('   Reading in dat {}'.format(i) + ' at ' + str(datetime.now()))
            newDat = pd.read_csv(path.format(i = str(i)),
                                 usecols = ['username', 'createdAtformatted', 'post', 'body'],
                                 parse_dates = ['createdAtformatted'])
            newDat = newDat[~pd.isnull(newDat.body)]
            
            # it should be the case that no users listed as joining after the cutoff date
            # posted anything before the cutoff date
            if not np.all(newDat.createdAtformatted[newDat.post == True].dt.date >= cutoffDate):
                newProblemPosts = newDat[(newDat.post == True)& (newDat.createdAtformatted.dt.date < cutoffDate)]
                problemPosts = pd.concat([problemPosts, newProblemPosts])
                
            
            # get creation date from creation datetime
            newDat['createdAtDate'] = newDat['createdAtformatted'].dt.date
            newDat['createdAtYear'] = newDat['createdAtformatted'].dt.year
            newDat['createdAtWeek'] = newDat['createdAtformatted'].dt.isocalendar().week
            newPlotDat = pd.DataFrame(columns = ['post', 'createdAtDate', 'createdAtWeek', 'createdAtYear'])

            # create a flag for each keys being true
            for key in keys:
                newDat = createTermFlags(newDat, terms = termsDict[key], key = key)
                newPlotDat = newPlotDat.merge(prepPlotDat(newDat, key), 
                                              on = ['post', 'createdAtDate', 'createdAtWeek', 'createdAtYear'], 
                                              how = 'outer', 
                                              validate = '1:1')
        
            # create a flag for any key being true, keep only this data
            newDat['flag'] =  newDat[[i + 'Any' for i in keys]].max(axis = 1)
            newDat = newDat[newDat.flag == True]
            plotDat = pd.concat([plotDat, newPlotDat])
            dat = pd.concat([dat, newDat])
            
        except:
            errors.append(i)
    
    plotDat = plotDat.reset_index().groupby(['post', 'createdAtDate', 'createdAtWeek', 'createdAtYear']).sum()
    dat.to_csv(os.path.join(outputPathDate, 'dat.csv'))
    plotDat.to_csv(os.path.join(outputPathDate, 'plotDat.csv'))
    problemPosts.to_csv(os.path.join(outputPathDate, 'problemPosts.csv'))
    pd.Series(errors).to_csv(os.path.join(outputPathDate, 'errors.csv'))
    
    return dat, plotDat

# plots need dat to be in a slightly different format
# by dat
def prepPlotDat(dat, key):
    plotDat = dat.groupby(['post', 'createdAtDate', 'createdAtWeek', 'createdAtYear']).agg({key + 'Any': ['count', 'sum']})[key + 'Any']
    plotDat.columns = [i + '_' + key for i in plotDat.columns]
    return plotDat.reset_index()


def createTermFlags(dat, terms, key):
    # iterate through terms, find which parleys contain them
    for term in terms:
        dat[term] = dat['body'].str.contains(term, regex = True, 
                                               flags = re.IGNORECASE)
    # create a flag for any of the terms mentioned
    dat[key + 'Any'] = dat[terms].max(axis = 1)

    return dat

def createCooccMat(dat, terms, outpath):
    # co-occurrence matrix
    mat = dat.loc[~pd.isnull(dat.body), terms].astype(int)
    coocc = mat.T.dot(mat)
    coocc.to_csv(os.path.join(outpath, 'coocc.csv'))
    return coocc

# lists a set of example posts for 
def examplePosts(dat, outpath):
    exampleRows = np.random.randint(low = 0, high = dat.size[0], size = 5)
    examples = dat.body.iloc[exampleRows]
    examples.to_csv(os.path.join(outpath, 'example.csv'))

# find other terms which frequently co-occur with the terms of interest
def findOtherTerms(dat, outpath):
    flatTokens, tokens = cleanText(dat)
    wordCount = flatTokens.value_counts()
    wordCount.to_csv(os.path.join(outpath, 'unigrams.csv'))
    bigramsCount = findBigrams(tokens, outpath)
    return wordCount, bigramsCount

def cleanText(dat):
    # strip punctuation
    texts = dat.body.str.replace('[{}]'.format(string.punctuation), '', regex = True).str.lower()
    # tokenise
    tokenizer = TweetTokenizer()
    tokens = [tokenizer.tokenize(i) for i in texts]
    tokensClean = [[i for i in j if i not in stopwords.words('english')] for j in tokens]
    flatTokens = pd.Series(np.concatenate(tokensClean))
    return flatTokens, tokens

def findBigrams(tokens, outpath):
    bigrams = [list(ngrams(token, 2)) for token in tokens]
    bigramsFlat = pd.Series([item for bigram in bigrams for item in bigram])
    bigramsCount = bigramsFlat.value_counts()
    bigramsCount.to_csv(os.path.join(outpath, 'bigrams.csv'))
    return bigramsCount

# creates plots of mentions through time for a specific topic
def makePlots(plotDat, key, outpath, cutoffDate):
    plotDat['createdAtDate'] = pd.to_datetime(plotDat.createdAtDate)
    plotDat['timeCounter'] = pd.Series(plotDat.createdAtDate - cutoffDate).dt.days.astype(int)
    plotDat['weekCounter'] = np.floor(plotDat['timeCounter'] / 7)
    plotDatAgg = plotDat.groupby(['weekCounter', 'post'], as_index = False).agg({
        'createdAtDate':'count',
        'count_{}'.format(key): 'sum',
        'sum_{}'.format(key): 'sum'
    })
    plotDatAgg['weekMean'] = plotDatAgg['sum_{}'.format(key)] / plotDatAgg['count_{}'.format(key)]
    plotDatAgg['weekObs'] = plotDatAgg['sum_{}'.format(key)]
    plotDatAgg['weekSD'] = plotDatAgg['weekMean'] * (1 - plotDatAgg['weekMean']) / np.sqrt(plotDatAgg['weekObs'])
    plotDatAgg['weekCI'] = plotDatAgg['weekSD'] * 1.96 * 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 7))
    ax1.errorbar(x = plotDatAgg.weekCounter[plotDatAgg.post], 
                 y = plotDatAgg.weekMean[plotDatAgg.post], 
                 yerr = plotDatAgg.weekCI[plotDatAgg.post],
                color = 'orange', alpha = 0.7)
    ax1.errorbar(x = plotDatAgg.weekCounter[~plotDatAgg.post], 
                 y = plotDatAgg.weekMean[~plotDatAgg.post], 
                 yerr = plotDatAgg.weekCI[~plotDatAgg.post],
                color = 'blue', alpha = 0.3)
    sns.lineplot(ax = ax1,
                 ci = 'sd',
                 x = 'weekCounter', y = 'weekMean', hue = 'post',
                data = plotDatAgg)
    sns.lineplot(ax = ax2,
                 x = 'weekCounter', y = 'weekObs', hue = 'post',
                data = plotDatAgg)
    ax1.legend(title = "User joined\nafter cutoff")
    ax2.legend(title = "User joined\nafter cutoff")
    ax1.set_xlabel('Weeks relative to cutoff date')
    ax1.set_ylabel('Proportion of parleys which mention {}'.format(key))
    ax2.set_xlabel('Weeks relative to cutoff date')
    ax2.set_ylabel('Number of parleys which mention {}'.format(key))
    
    fig.savefig(os.path.join(outpath, 'fig.png'))
    
    
    # get number/average mentions over time
#    forPlotSum = dat.pivot('createdAtDate', 'post', 'sum' + '_' + key).fillna(0)
#    dat['m'] = dat['sum'+ '_' + key]/dat['count'+ '_' + key]
#    forPlotMean = dat.pivot('createdAtDate', 'post', 'm').fillna(0)
    # make plot
#    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 7))
#    sns.lineplot(ax = ax1, data = forPlotMean)
#    sns.lineplot(ax = ax2,data = forPlotSum)
#    ax1.legend(title = "User joined\nafter cutoff")
#    ax2.legend(title = "User joined\nafter cutoff")
#    fig.savefig(outpath + 'fig.png')
#    return fig

# performs analysis for a specific topic
def summariseParleys(key, terms, dat, plotDat, outpath, cutoffDate):
    dat = createTermFlags(dat, terms, outpath)
    fig = makePlots(plotDat, key, outpath, cutoffDate)
    cooccMat = createCooccMat(dat, terms, outpath)
    wordCount, bigramsCount = findOtherTerms(dat, outpath)
    testDiff(plotDat, key, outpath)
    return cooccMat, wordCount, bigramsCount, fig

# checks to see if a folder for the output exists, if not creates one,
# then returns its path
def createFolder(outpath, new):
    outpathTopic = os.path.join(outpath, new)
    if not os.path.exists(outpathTopic):
        os.mkdir(outpathTopic)
    return outpathTopic

def testDiff(plotDat, key, outpath):
    
    # create the proportion of all posts that contain the term
    propAggAllPosts = plotDat.reset_index().groupby('post')\
        .agg({'count_' + key: np.sum, 'sum_' + key: np.sum})
    # do a z-test for proportions
    allPostsTest = proportions_ztest(count = propAggAllPosts['sum_' + key], 
                                     nobs = propAggAllPosts['count_' + key])
    # create the proportion of users who have any posts that contain the term
    #propAggAnyPosts = file1.groupby(['username', 'post'])\
    #    .agg({'count': np.sum, 'flag': np.max}).groupby('post')\
    #    .agg({'count': np.sum, 'flag': np.sum})
    # z test for proportions
    #anyPostsTest = proportions_ztest(count = propAggAnyPosts['flag'], 
    #                                nobs = propAggAnyPosts['count'])

    # make table
    meanPre = np.round(propAggAllPosts.iloc[0,:]['sum_' + key]/propAggAllPosts.iloc[0,:]['count_' + key], 3)
              
    meanPost = np.round(propAggAllPosts.iloc[1,:]['sum_' + key]/propAggAllPosts.iloc[1,:]['count_' + key], 3)
              
    
    stats = np.round(allPostsTest[0],3) 
    pvals = np.round(allPostsTest[1], 3)
    answer = pd.DataFrame(index = ['By post'],
                         data = {'Mean pre': meanPre, 'Mean post': meanPost,
                                 'Z': stats, 'p': pvals})
    answer.to_csv(os.path.join(outpath, 'f.csv'))
    return(answer)

# performs the analysis for the group of topics fed in
def runTopicFinder(termsDict, inpath = c.samplePostsPath,
                   outpath = c.topicsAnalysisPath):
    keys = termsDict.keys()
    for i in range (c.dates.shape[0]):
        
        cutoffDate = pd.Timestamp(c.dates.loc[i, 'years'],
                                 c.dates.loc[i, 'months'],
                                 c.dates.loc[i, 'days'])
        print(cutoffDate)
        outpathDate = createFolder(outpath, 'sample' + str(cutoffDate.date()))
        inpathDate = inpath.format(date = 'sample' + str(cutoffDate.date()), i = '{i}')
        print(inpathDate)
        dat, plotDat = dataIn(inpathDate, termsDict,
                              keys, cutoffDate, outpathDate)
        for key in keys:
            print(key)
            print(datetime.now())
            thisDat = dat.loc[dat[(key + 'Any')] == True,:]
            outpathTopic = createFolder(outpathDate, key)
            summariseParleys(key, termsDict[key], thisDat, 
                 plotDat.loc[:,plotDat.columns.str.contains(key)].reset_index(), 
                 outpathTopic,
                 cutoffDate)
        
if __name__ == '__main__':
    runTopicFinder(c.termsDict)