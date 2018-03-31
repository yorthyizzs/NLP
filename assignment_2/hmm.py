import re
import constants
from collections import Counter

class HiddenMarkovModel:
    def __init__(self, datafile):
        self.datafile = datafile
        self.unilines = []
        self.mispelled = []
        self.words = []
        self.raw_words = []
        self.lines = []
        self.unigram = None
        self.bigram = None
        self.deletion = []
        self.insertion = []
        self.substitution = []
        self.morphemes = set()
        self.insertProbs = {}
        self.delProbs = {}
        self.subProbs = {}
        self.misvocab = {}
        self.morphemeCount = {}
        self.mispelledCorrection = []

    def _readData(self):
        fp = open(self.datafile)
        return fp.readlines()

    def _collectData(self):
        for line in self._readData():
            self.unilines.append(self._correctLine(line.strip()))

    # correct the error word with the target ones
    def _correctLine(self, line):
        errors = re.finditer(constants.ERROR_REGEX, line)
        for error in errors:
            line = line.replace(error.group(0), error.group(1)) # correct the sentences to get clear data
            self.mispelled.append(error.group(2)) # collect mispelled words
        return line

    def _extractWords(self):
        for line in self.unilines:
            line = re.split(constants.WORDS_EXTRACTER, line) # extract the words and punctuations
            # take only the words and make them lower case
            line = [w.lower() for w in line if  (w != '' and w != ' ' and
                                                 w != '\n' and w != '"' and
                                                 w != '.' and w != ',' and
                                                 w != '!' and w != '?' and
                                                 w != ';' and w != '/' and
                                                 w != '\\')]
            if line:
                self.raw_words.extend(line) # collect the words at the same time
                line.insert(0, constants.START) # add sentence boundaries to each line
                line.append(constants.END)
                self.lines.append(line) # collect the modified sentences
        self.unigram = Counter(self.raw_words) # count the word occurences
        self.words = set(self.raw_words) # hold only unique words
        self.mispelled = set(self.mispelled)

    def  _buildBigram(self):
        tempbigram = []
        for line in self.lines:
            for i in range(1, len(line)):
                tempbigram.append(line[i-1] + ' ' + line[i])
        self.bigram = Counter(tempbigram)

    def _calculateEdits(self):
        for mispelled in self.mispelled:
            mispelled = mispelled.lower()
            self.misvocab[mispelled] = []
            # check word's substitution / insertion / deletion words
            for word in self.words:
                if len(word) == len(mispelled):
                    self._checkSubstituion(mispelled, word)
                elif len(word) - len(mispelled) == 1:
                    self._checkDeletion(mispelled, word)
                elif len(mispelled) - len(word) == 1:
                    self._checkInsertion(mispelled, word)

            # if there is a mispelled 2 word then check the words with edit distance 1 for it too
            if len(mispelled.split(' ')) > 2:
                for line in self.lines:
                    for i in range(2, len(line)-1):
                        word = line[i-1]+' '+line[i]
                        if len(word) == len(mispelled):
                            self._checkSubstituion(mispelled, word)
                        elif len(word) - len(mispelled) == 1:
                            self._checkDeletion(mispelled, word)
                        elif len(mispelled) - len(word) == 1:
                            self._checkInsertion(mispelled, word)

        # count the occurences of those edit operations
        self.deletion = Counter(self.deletion)
        self.insertion = Counter(self.insertion)
        self.substitution = Counter(self.substitution)

    # if the words are in same length check if there is only one edit if so add to corresponding indexes
    def _checkSubstituion(self, word, candidate):
        temp_sub = None
        temp_divider = None
        for i in range(len(word)):
            if word[i] != candidate[i]:
                if temp_sub is None:
                    temp_sub = candidate[i]+''+word[i]
                    temp_divider = candidate[i]
                else:
                    return
        if temp_sub:
            self.morphemes.add(temp_divider)
            self.substitution.append(temp_sub)
            self.subProbs[word + ' ' +candidate] = temp_sub
            self.misvocab[word].append(candidate)

    # if there might be deletion probability between the words
    # check if one deletion operation is ok to convert them to each other
    # and add the operation to corresponding indexes
    def _checkDeletion(self, word, candidate):
        temp_del = None
        for i in range(len(word)):
            if word[i] != candidate[i]: # if alignment broke
                if i == 0: temp_del= '#'+candidate[i]
                else: temp_del= candidate[i-1]+''+candidate[i]  # set as edit
                for j in range(i, len(word)): # check if there is another edit
                    # if so distance!=1 return
                    if word[j] != candidate[j+1]: return
                break
        if temp_del is None: # if alignment did not break it means last character is missed
            temp_del = candidate[len(candidate)-2] + '' + candidate[len(candidate)-1]
        self.morphemes.add(temp_del)
        self.deletion.append(temp_del)
        self.delProbs[word + ' ' +candidate] = temp_del
        self.misvocab[word].append(candidate)

    # if there might be insertion probability between the words
    # check if one insertion operation is ok to convert them to each other
    # and add the operation to corresponding indexes
    def _checkInsertion(self, word, candidate):
        temp_ins = None
        temp_divider = None
        for i in range(len(candidate)):
            if word[i] != candidate[i]:  # if alignment broke
                if i == 0:
                    temp_ins = ' ' + word[i]
                    temp_divider = ' '
                else:
                    temp_ins = word[i - 1] + '' + word[i]  # set as edit
                    temp_divider = word[i - 1]
                for j in range(i, len(candidate)):  # check if there is another edit
                    # if so distance!=1 return
                    if candidate[j] != word[j + 1]:
                        return
                break
        if temp_ins is None:  # if alignment did not break it means last character is missed
            temp_ins = word[len(word) - 2] + '' + word[len(word) - 1]
            temp_divider = word[len(word) - 2]
        self.morphemes.add(temp_divider)
        self.insertion.append(temp_ins)
        self.insertProbs[word + ' ' + candidate] = temp_ins
        self.misvocab[word].append(candidate)

    # the morphemes that is calculated in operation calculation
    # which will be needed for emission probabilities
    # will be count here
    def _countMorphemes(self):
        bigStr = ''
        for line in self.unilines:
            bigStr+=line.lower()
        for morpheme in self.morphemes:
            if morpheme[0] == '#':
                count = 0
                for word in self.raw_words:
                    if word[0] == morpheme[1]: count += 1
                self.morphemeCount[morpheme] = count
            else:
                self.morphemeCount[morpheme] = bigStr.count(morpheme)

    # for viterbi calculation return the bigram probability of corresponding two words in a row
    # a.k.a transission probability
    def getBigramProb(self, word1, word2):
        divident = 1
        divider = len(self.unigram.keys())
        bigram = word1 + ' ' + word2
        if bigram in self.bigram.keys():
            divident += self.bigram[bigram]
        if word1 in self.unigram.keys():
            divider += self.unigram[word1]
        return float(divident/divider)

    # for viterbi calculation return the edit probability of between two word
    # a.k.a emission probability
    def getEditProb(self, word1, word2):
        key = word1 + ' ' + word2
        divident = 0
        divider = 1
        if key in self.insertProbs.keys():
            inner_key = self.insertProbs[key]
            divident = self.insertion[inner_key]
            divider = self.morphemeCount[inner_key[0]]
        if key in self.delProbs.keys():
            inner_key = self.delProbs[key]
            divident = self.deletion[inner_key]
            divider = self.morphemeCount[inner_key]
        if key in self.subProbs.keys():
            inner_key = self.subProbs[key]
            divident = self.substitution[inner_key]
            divider = self.morphemeCount[inner_key[0]]

        return float(divident/divider)

    # return if corresponding word has a candidate word
    def inVocab(self, mispelled):
        return len(self.misvocab[mispelled]) != 0


    def train(self):
        self._collectData()
        self._extractWords()
        self._buildBigram()
        self._calculateEdits()
        self._countMorphemes()
       




