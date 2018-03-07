import numpy as np
import random
import constants

class Bigram:
    def __init__(self, sentences, words, smooth=False):
        self.smooth = smooth
        self.sentences = sentences
        self.words = np.array(list(words))
        self.wordCounts = None
        self.counts = np.zeros((len(words), len(words)))
        self.probs = np.zeros((len(words), len(words)))

    def countTheWords(self):
        for sentence in self.sentences:
            for i in range(len(sentence)-1):
                x = np.argwhere(self.words == sentence[i])[0]#previous word
                y = np.argwhere(self.words == sentence[i+1])[0]#input word
                # and y is the word that comes before the x
                self.counts[x, y] += 1

    def calculateTheProbs(self):
        if self.smooth:
            add_one = self.counts+1
            divider = np.sum(add_one, axis=1)
            self.probs = add_one / divider
        else:
            divider = np.sum(self.counts, axis=1)
            self.probs = self.counts / divider
            self.probs[np.isnan(self.probs)] = 0

    def generateMail(self):
        wordCount = 0
        log_probs = np.log2(self.probs)  # in the mean time calculate perplexity
        perplexity = 0

        #FIRST ATTEMPT######
        wordIndex1 = np.argwhere(self.words == constants.sentence_begin)[0][0]#set word1 as <s> to start sentence
        rand_prob = random.uniform(0, np.max(self.probs[wordIndex1]))
        columnOfword1 = self.probs[wordIndex1].reshape(len(self.words))
        minOfMax = np.min(columnOfword1[columnOfword1 >= rand_prob])
        wordIndex2 = np.argwhere(columnOfword1 == minOfMax)[0][0]
        word2 = self.words[wordIndex2]
        sentence, stop = self._addWordToSentence(word2, '')
        wordCount += 1
        perplexity += log_probs[wordIndex1,wordIndex2]

        while wordCount < 31 and not stop:  # stop if there is 30 words or stop flag is opened
            wordIndex1 = wordIndex2
            rand_prob = random.uniform(0, np.max(self.probs[wordIndex1]))
            columnOfword1 = self.probs[wordIndex1].reshape(len(self.words))
            minOfMax = np.min(columnOfword1[columnOfword1 >= rand_prob])
            wordIndex2 = np.argwhere(columnOfword1 == minOfMax)[0][0]
            word2 = self.words[wordIndex2]
            perplexity += log_probs[wordIndex1, wordIndex2]
            sentence, stop = self._addWordToSentence(word2, sentence)
            wordCount += 1
        perplexity = 2 ** (-1 / wordCount * perplexity)
        return sentence, perplexity

    def _addWordToSentence(self, w, sentence):
        if w == constants.sentence_end:
            return sentence, True
        if w == constants.sentence_begin:
            return sentence, False
        return sentence + w + ' ', False

