#  This module contains tools for getting features from a data file


from sklearn.feature_extraction.text import CountVectorizer

def readHamSpamFile(filename):
    features = []

    f = open(filename, 'r')
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
    raw_features = readHamSpamFile('train.csv')
    #print(raw_features)

#okay, that looks good.  Can we tokenize it???

def tokenizeRawFeatures(raw_features):

    vectorizer = CountVectorizer()
    corpus = [x[1] for x in raw_features]
    #print(corpus)
    X = vectorizer.fit_transform(corpus)
    print(X.toarray())
    
if __name__=='__main__':
    tokenizeRawFeatures(raw_features)
