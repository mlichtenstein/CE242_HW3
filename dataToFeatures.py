#  This module contains tools for getting features from a data file

print('importing CountVectorizer...')
from sklearn.feature_extraction.text import TfidfVectorizer
print('   imported')

#%% First, let's make a vectorizer:
def getVectorizerFromHamSpamFile(filename):
    '''Outputs (v, X, y):
        v = a vectorizer, initialized from hamspam data in filename
        X = a matrix, each row of which is a feature from original hamspam file
        y = a vector of booleans, for the ham/spam (spam = 1) labels
    '''
    corpus = []
    y = []
    vectorizer = TfidfVectorizer()
    
    #Read the file, get y and a vector of strings:
    f = open(filename, 'r')
    print('getVectorizer: reading data from ' + filename + '...')
    for line in f:
        try:
            (hamspam, subjectline) = line.split(',',1)
            corpus.append( subjectline ) 
            y.append(hamspam=='spam',)
        except Exception as e:
            continue
            print('getVectorizer:  error parsing a line:', line, e)
            input('...')
    f.close()
    
    #and now do the vectorizer thing:
    X = vectorizer.fit_transform(corpus)
    
    return(vectorizer, X, y)
    
if __name__=='__main__':
    print('\ntesting getVectorizer:...')
    (vec, X, y) = getVectorizerFromHamSpamFile('train1.csv')
    print(vec.get_feature_names())
    
#%% Cool, let's see if we can get features from a different dataset this way:
def useVectorizerOnHamSpamFile(vectorizer,filename):
    '''outputs:  (X, y) 
    X = matrix of feature vectors 
    y = vector of booleans for labels
    '''
    corpus = []
    y = []
    
    f = open(filename, 'r')
    print('useVectorizer:  reading data from ' + filename + '...')
    for line in f:
        try:
            (hamspam, subjectline) = line.split(',',1)
            corpus.append( subjectline ) 
            y.append(hamspam=='spam',)
        except Exception as e:
            continue
            print('useVectorizer:  error parsing a line:', line, e)
            input('...')
    f.close()
    
    # get features:
    X = vectorizer.transform(corpus)
    return(X,y)
    
if __name__=='__main__':
    print('\ntesting useVectorizer:..')
    (X2, y2) = useVectorizerOnHamSpamFile(vec, 'train2.csv')
    print(X2)
    
    #cool, looks good!


#%% Finally, let's save the results:
import pickle

if __name__=='__main__':
    output = open('protoFeatures.pkl', 'wb')
    for data in [vec, X, y, X2, y2]:
        pickle.dump(data,output)
    output.close()