import pandas as pd
SEED = 8338
# the number of files in the posts, users samples
NPOSTS = 166
NUSERS = 8


wd = '/home/beeb/Insync/sei2112@columbia.edu/Google Drive/Columbia SusDev PhD/Research/My work/parler/'

# code for assembling raw 
partsPath = 'data/intermediate/parlerText/inParts/'
wholePath = 'data/intermediate/parlerText/whole/whole.csv'
assemblyHandlingPath = 'data/intermediate/parlerText/handling/'
partialTextPath = 'data/raw/parler_2020-01-06_posts-partial/allfiles/'

userInpath = 'data/raw/parler/users/parler_user00000000000{}.ndjson'
userOutpath = 'data/intermediate/parler/users/all/parler_user00000000000{}.csv'

sampleUsersPath = 'data/intermediate/parler/users/sample/userSample{}.csv'
samplePostsPath = 'data/intermediate/parler/posts/sample/{date}/postSample{i}.csv'

keyPath = '~/Documents/keys/Administrator_accessKeys.csv'

topicsAnalysisPath = 'output/findTopics/'

# these variables set up the creation of the sample for the RDD
dates = pd.DataFrame({'years': [2020],#[2018, 2019, 2020, 2020],
               'months': [6],#[12, 6, 6, 11],
               'days': [16]})#[9, 1, 16, 4],
               #'desc': ['Candace Owens tweets', 'Saudis', 
               #         'Dan Bongino announces purchase of ownership stake on Parler',
               #        'Election']})
nDaysBefore = 180
nDaysAfter = 90

# this lists the terms that are used to identify different topics
termsDict = {'qanon': ['qanon', 'wwg', 'wga', 'areqawake', 'qarmy', 'the[ ]{0,1}great[ ]{0,1}awakening',
        'the[ ]{0,1}storm', 'where[ ]{0,1}we[ ]{0,1}go[ ]{0,1}one[ ]{0,1}we[ ]{0,1}go[ ]{0,1}all',
        'trust[ ]{0,1}the[ ]{0,1}plan'],
            'health': ['vacc', 'covid', 'virus'],
            'china': ['china']}