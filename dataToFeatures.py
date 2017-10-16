#  This module contains tools for getting features from a data file

print('importing CountVectorizer...')
from sklearn.feature_extraction.text import CountVectorizer

def readHamSpamFile(filename):
    features = []

    f = open(filename, 'r')
    print('reading data from ' + filename + '...')
    for line in f:
        try:
            (hamspam, subjectline) = line.split(',',1)
            features.append( (
                hamspam=='ham',
                subjectline) )
        except Exception as e:
            continue
            print(line)
            print(e)
            input('...')
    f.close()
    return features

if __name__=='__main__':
    print('\nbeginning dataToFeatures test...')
    raw_features = readHamSpamFile('train1.csv')
    print('got %d features' % (len(raw_features)))
    print('here are the raw features 1:')
    print(raw_features)

#okay, that looks good.  Can we tokenize it???

def tokenizeRawFeatures(raw_features, vectorizer = None):
    '''A function that takes raw features( a list of tuples, (bool, string))
    and optionally an extant vectorizer.
    and returns (X, vectorizer)
    where X is the set of tokenized features
    and vectorizer is the vectorizer that produced that.'''

    #if no vectorizer is specified, create a new vectorizer:
    if vectorizer == None:
        print('creating a new vectorizer...')
        vectorizer = CountVectorizer()

    corpus = [x[1] for x in raw_features]
    X = vectorizer.fit_transform(corpus)
    return(X, vectorizer)

if __name__=='__main__':
    print('training raw features on a new vectorizer:')
    X,vectorizer = tokenizeRawFeatures(raw_features)
    print('here is the vectorized corpus 1:')
    print(X)
    print('here is the vocabulary of the vectorizer:')
    print(vectorizer.vocabulary_)


    #we might also want to run this on a different set of features, so let's see how that looks:

    raw_features2 = readHamSpamFile('train2.csv')
    X2 = tokenizeRawFeatures(raw_features2,vectorizer)

    print('here are the raw features 2:')
    print(raw_features)
    print('here is the vectorized corpus 2:')
    print(X)