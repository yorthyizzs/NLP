import re
from collections import Counter
import constants
import random
import math
import re

class Model:
    def __init__(self, sentences, test_data):
        self.sentences = sentences
        self.unigram = None
        self.bigram = None
        self.trigram = None
        self.testmails = test_data

    def train(self):
        self._createUnigram()
        self._createBigram()
        self._createTrigram()

    def generateUnigramMail(self, smooth=False):
        wordCount = 0
        word = ''
        mail = ''
        perplexity = 0
        probability = 1
        divider = sum(self.unigram.values())
        # generate word till you reach 30 word or generate stop punctuations
        while wordCount <= 30 and word != '.' and word != '!' and word != '?':
            word, prob = self._findRandomized(self.unigram, divider, smooth) # get randomize word and it's probability
            mail += ' ' + word # add word to ultimate sentence
            wordCount += 1
            perplexity += math.log(prob, 2) # in the mean time calculate perplexity by using words probabilies
            probability *= prob # calculate probabilies too
        return mail, 2**(-1*perplexity/wordCount), probability # return generated mail, it's perplexity and probability

    def generateBigramMail(self, smooth=False):
        wordCount = 0
        word = constants.sentence_begin
        mail = ''
        perplexity = 0
        probability = 1
        # generate word till reach 30 word or generate sentence end
        while wordCount <= 30 and word != constants.sentence_end:
            subDict = self._subDict(word, self.bigram) # create a subdict from the keys that start with previous word
            divider = sum(subDict.values()) # to find denominator of subset sum all values
            word, prob = self._findRandomized(subDict, divider, smooth) #get the generated word and it's probability
            word = word.split()[1] #take the new added word
            if word != constants.sentence_end and word != constants.sentence_begin: #if it is a real word add to ultimate sentence
                mail += ' '+word
            wordCount += 1
            perplexity += math.log(prob,2) #calculate perplexity in meantime
            probability *= prob #also probability
        return mail, 2**(-1*perplexity/wordCount), probability

    def generateTrigramMail(self, smooth=False):
        wordCount = 0
        word2 = constants.sentence_begin
        word3 = constants.sentence_begin
        mail = ''
        perplexity = 0
        probability = 1
        # generate word till reach 30 word or generate sentence end
        while wordCount <= 30 and word3 != constants.sentence_end:
            word = word2 + ' ' + word3
            subDict = self._subDict(word, self.trigram) # create a subdict from the keys that starts with previously 2 words
            divider = sum(subDict.values()) # find the denominator of subset
            word, prob = self._findRandomized(subDict, divider, smooth) #get generated 3 word and it's probability
            [_, word2, word3] = word.split() #get next iteration previously 2 words
            if word3 != constants.sentence_end and word3 != constants.sentence_begin:
                mail += ' ' + word3 # add the new word
            wordCount += 1
            perplexity += math.log(prob, 2)# calculate perplexity in mean time
            probability *= prob # and probability also
        return mail, 2 ** (-1 * perplexity / wordCount), probability

    def calculateTestProbs(self):
        # this method simply take test emails and calculate their probabilities
        # by using the smoothed trigram model.
        testProbs = []
        for mail in self.testmails:
            mailBody = []
            sentences = re.split(constants.sentence_regex, mail) #split corresponding mail's sentences
            for sentence in sentences:
                splitted = (re.split('(\)|\(|/|\?|!|\.|,|;|:|\n|\s)', sentence)) #split sentence to words
                splitted = [w for w in splitted if (w != '' and w != ' ' and w != '\n')]
                if splitted : mailBody.extend(splitted)
            testProbs.append(self._testProbHelper(mailBody)) # calculate mail probability and add to list
        return testProbs

    def _testProbHelper(self, words):
        # this method is a helper method to test data calculation method
        log_prob = 0
        divider = sum(self.trigram.values()) + len(self.trigram.keys()) # find the denominator (smoothed)
        for i in range(len(words)-2):
            temp = words[i] + ' ' + words[i+1] + ' ' + words[i+2] # take trigram of sentence
            numerator = self.trigram[temp] if temp in self.trigram.keys() else 0
            # take frequency if trigram exist in model else it's 0
            log_prob += math.log(2, (numerator+1)/divider)
        return log_prob


    def _addSentenceBoundaries(self):
        # this methods add <s> and <\s> to sentence as a boundary
        for sentence in self.sentences:
            sentence.insert(0, constants.sentence_begin)
            sentence.append(constants.sentence_end)

    def _createUnigram(self):
        # this method also create sentences array
        # which contains it's words as a list in it.(splittedSentences[])
        # it simply split sentences and counts the words
        wordList = []
        splittedSentences = []
        for sentence in self.sentences:
            splitted = (re.split('(\)|\(|/|\?|!|\.|,|;|:|\n|\s)', sentence))
            splitted = [w for w in splitted if (w != '' and w != ' ' and w != '\n')]
            if splitted:
                wordList.extend(splitted)
                splittedSentences.append(splitted)
        self.unigram = Counter(wordList)
        self.sentences = splittedSentences

    def _createBigram(self):
        self._addSentenceBoundaries()
        # add <s> and <\s> to sentences as it will necessary in bigram
        bigramWordList = []
        # takes the two grouped words (bigrams) and count them in the end
        for sentence in self.sentences:
            for i in range(len(sentence)-1):
                bigram = sentence[i] + ' ' + sentence[i+1]
                bigramWordList.append(bigram)
        self.bigram = Counter(bigramWordList)

    def _createTrigram(self):
        self._addSentenceBoundaries()
        # put one more boundary as it necessary it trigram (ex: "<s> <s> I")
        trigramWordList = []
        # take 3 grouped words as a trigram and counts them in the end
        for sentence in self.sentences:
            for i in range(len(sentence)-2):
                trigram = sentence[i] + ' ' + sentence[i+1] + ' ' + sentence[i+2]
                trigramWordList.append(trigram)
        self.trigram = Counter(trigramWordList)

    def _subDict(self, starterKey, ngram):
        # this method creates a sub dict with the keys that starts with given key
        return {key: val for key, val in ngram.items() if key.startswith(starterKey+' ')}

    def _findRandomized(self, ngram, divider, smooth=False):
        # this method find randomized word with smoothed and unsmoothed way
        prob = 0
        rand = random.uniform(0, 1)
        for key in ngram:
            if smooth:
                prob1 = ((ngram[key] + 1) / (divider + len(ngram.keys())))
                prob += prob1
            else:
                prob1 = (ngram[key] / divider)
                prob += prob1
            if prob > rand:
                return key, prob1







