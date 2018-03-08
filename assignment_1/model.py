import re
from collections import Counter
import constants
import random
import math

class Model:
    def __init__(self, sentences):
        self.sentences = sentences
        self.unigram = None
        self.bigram = None
        self.trigram = None

    def train(self):
        self._createUnigram()
        self._createBigram()
        self._createTrigram()

    def generateUnigramMail(self, smooth=False):
        wordCount = 0
        word = ''
        mail = ''
        perplexity = 0
        divider = sum(self.unigram.values())
        while wordCount <= 30 and word != '.' and word != '!' and word != '?':
            word, prob = self._findRandomized(self.unigram, divider, smooth)
            mail += ' ' + word
            wordCount += 1
            perplexity += math.log(prob, 2)
        return mail, 2**(-1*perplexity/wordCount)

    def generateBigramMail(self, smooth=False):
        wordCount = 0
        word = constants.sentence_begin
        mail = ''
        perplexity = 0
        while wordCount <= 30 and word != constants.sentence_end:
            subDict = self._subDict(word, self.bigram)
            divider = sum(subDict.values())
            word, prob = self._findRandomized(subDict, divider, smooth)
            word = word.split()[1]
            if word != constants.sentence_end and word != constants.sentence_begin:
                mail += ' '+word
            wordCount += 1
            perplexity += math.log(prob,2)
        return mail, 2**(-1*perplexity/wordCount)

    def generateTrigramMail(self, smooth=False):
        wordCount = 0
        word1 = constants.sentence_begin
        word2 = constants.sentence_begin
        word3 = constants.sentence_begin
        mail = ''
        perplexity = 0
        while wordCount <= 30 and word3 != constants.sentence_end:
            word = word1 + ' ' + word2
            subDict = self._subDict(word, self.trigram)
            divider = sum(subDict.values())
            word, prob = self._findRandomized(subDict, divider, smooth)
            [word1, word2, word3] = word.split()
            if word3 != constants.sentence_end and word3 != constants.sentence_begin:
                mail += ' ' + word3
            wordCount += 1
            perplexity += math.log(prob, 2)
        return mail, 2 ** (-1 * perplexity / wordCount)

    def _addSentenceBoundaries(self):
        for sentence in self.sentences:
            sentence.insert(0, constants.sentence_begin)
            sentence.append(constants.sentence_end)

    def _createUnigram(self):
        # this method also create sentences array
        # which contains it's words as a list in it.(splittedSentences[])
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
        for sentence in self.sentences:
            for i in range(len(sentence)-1):
                bigram = sentence[i] + ' ' + sentence[i+1]
                bigramWordList.append(bigram)
        self.bigram = Counter(bigramWordList)

    def _createTrigram(self):
        self._addSentenceBoundaries()
        # put one more boundary as it necessary it trigram (ex: "<s> <s> I")
        trigramWordList = []
        for sentence in self.sentences:
            for i in range(len(sentence)-2):
                trigram = sentence[i] + ' ' + sentence[i+1] + ' ' + sentence[i+2]
                trigramWordList.append(trigram)
        self.trigram = Counter(trigramWordList)

    def _subDict(self, starterKey, ngram):
        return {key: val for key, val in ngram.items() if key.startswith(starterKey)}

    def _findRandomized(self, ngram, divider, smooth=False):
        prob = 0
        key = None
        rand = random.uniform(0, 1)
        for key in ngram:
            if smooth:
                prob += ((ngram[key] + 1) / (divider + len(ngram.keys())))
            else:
                prob += (ngram[key] / divider)
            if prob > rand:
                break
        return key, prob







