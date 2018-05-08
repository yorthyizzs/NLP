import numpy as np
import re


WORD_REGEX = r'(\)|\(|/|\?|!|\.|,|;|:|\n|"|\s)'

# create a dictionary that has vector representation of words
def createWord2Vec(vec):
    vector_fp = open(vec, 'r')
    word2vec = {}
    for line in vector_fp.readlines():
        word, vec = line.split(':')
        vec = vec.split()
        vec = [float(i) for i in vec]
        word2vec[word.strip()] = np.array(vec)
    return word2vec

def createFeatureAndLabels(file, label, word2vec):
    fp = open(file, 'r')
    label = [0,1] if label == 1 else [1,0] # if label is positive [0, 1] else [1,0]
    labels = []
    feats = []
    # split the sentences into words and create their vector representations
    for line in fp.readlines():
        word_arr = re.split(WORD_REGEX, line)
        word_arr = [w.strip() for w in word_arr if (w != '' and w != ' ' and
                                                 w != '\n' and w != '"' and
                                                 w != '.' and w != ',' and
                                                 w != '!' and w != '?' and
                                                 w != ';' and w != '/' and
                                                 w != '\\')]
        labels.append(label)
        vec = np.zeros(200)
        for word in word_arr:
            if word in word2vec.keys():
                vec += word2vec[word]
        feats.append(normalize(vec))
    return np.array(feats), np.array(labels)

# I normalize the each vector it seemed it improve the accuracy results a bit.
def normalize(vec):
    maxs = np.max(vec)
    mins = np.min(vec)
    rng = maxs - mins
    return (vec-mins) / rng

def createInputs(pos, neg, vec):
    word2vec = createWord2Vec(vec)
    positive, plabels = createFeatureAndLabels(pos, 1, word2vec)
    negative, nlabels = createFeatureAndLabels(neg, 0, word2vec)
    # arrange the features and labels together
    features = np.concatenate((positive, negative))
    labels = np.concatenate((plabels, nlabels))
    # to shuffle create a index matrix and shuffle it
    index =np.arange(labels.shape[0])
    np.random.shuffle(index)
    #shuffle both labels and features according to index
    labels = labels[index,:]
    features = features[index,:]
    return features, labels


