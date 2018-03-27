import re
import constants
from collections import Counter


class HiddenMarkovModel:
    def __init__(self, datafile):
        self.datafile = datafile
        self.unilines = []
        self.mispelled = []
        self.words = []
        self.lines = []
        self.unigram = None
        self.bigram = None
        self.deletion = []
        self.insertion = []
        self.substitution = []
        self.morphemes = []

    def _readData(self):
        fp = open(self.datafile)
        return fp.readlines()

    def _correctData(self):
        for line in self._readData():
            self.unilines.append(self._correctLine(line.strip()))

    def _correctLine(self, line):
        errors = re.finditer(constants.ERROR_REGEX, line)
        for error in errors:
            line = line.replace(error.group(0), error.group(1)) # correct the sentences to get clear data
            self.mispelled.append(error.group(2)) # collect mispelled words
        return line

    def _extractWords(self):
        for line in self.unilines:
            line = re.split(constants.WORDS_EXTRACTER, line) # extract the words and punctuations
            line = [w for w in line if  (w != '' and w != ' ' and w != '\n')]
            if line:
                self.words.extend(line) # collect the words at the same time
                line.insert(0, constants.START) # add sentence boundaries to each line
                line.append(constants.END)
                self.lines.append(line) # collect the modified sentences
        self.unigram = Counter(self.words) # count the word occurences
        self.words = set(self.words) # hold only unique words
        self.mispelled = set(self.mispelled)

    def _buildBigram(self):
        tempbigram = []
        for line in self.lines:
            for i in range(1, len(line)):
                tempbigram.append(line[i-1]+ ' ' + line[i])
        self.bigram = Counter(tempbigram)

    def _calculateEdits(self):
        for mispelled in self.mispelled:
            for word in self.words:
                if len(word) == len(mispelled):
                    self._checkSubstituion(mispelled, word)
                elif len(word) > len(mispelled):
                    self._checkDeletion(mispelled, word)
                else:
                    print("")

    def _checkSubstituion(self, word, candidate):
        temp_sub = None
        temp_divider = None
        for i in range(len(word)):
            if word[i] != candidate[i] :
                if temp_sub is not None:
                    temp_sub = candidate[i]+' '+word[i]
                    temp_divider = word[i]
                else :
                    return
        self.morphemes.append(temp_divider)
        self.substitution.append(temp_sub)

    def _checkDeletion(self, word, candidate):
        temp_del = None
        for i in range(len(word)):
            if word[i] != candidate[i]: # if alignment broke
                if i == 0 : temp_del= '# '+candidate[i]
                else : temp_del= candidate[i-1]+' '+candidate[i]  # set as edit
                for j in range(i+1, len(word)): # check if there is another edit
                    # if so distance!=1 return
                    if word[j] != candidate[j+1]: return
                break
        if temp_del is None: # if alignment did not break it means last character is missed
            temp_del = candidate[len(candidate)-2] + ' ' + candidate[len(candidate)-1]
        self.morphemes.append(temp_del)
        self.deletion.append(temp_del)







    def train(self):
        self._correctData()
        self._extractWords()
        self._buildBigram()
        self._checkDeletion('acress', 'actress')
        self._checkDeletion('actres', 'actress')
        self._checkDeletion('ctress', 'actress')






