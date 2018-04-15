import re
import constants
import stemmer
from collections import Counter
import numpy as np


class Lexelt:
    def __init__(self, name):
        self.name = name
        self.senses = []
        self.numberofwords = 0
        self.sense_ids = set()

    def addSense(self, sense):
        self.senses.append(sense)
        self.sense_ids.add(sense.id)
        self.numberofwords += sense.numberofwords

    def getSenseContextWordNumber(self, senseid):
        number = 0
        for sense in self.senses:
            if sense.id == senseid:
                number += sense.numberofwords
        return number

    def getSenseWordNumber(self, word, senseid):
        number = 0
        for sense in self.senses:
            if sense.id == senseid:
                for key, val in sense.words.items():
                    if key == word:
                        number += val

        return number

    def getPrior(self, senseid):
        return float(self.getSenseContextWordNumber(senseid) / 800000)

    def getLikelihood(self, word, senseid):
        return float((self.getSenseWordNumber(word, senseid) + 1) / self.getSenseContextWordNumber(senseid))


class Sense:
    def __init__(self, id, context):
        self.id = id
        self.feature = []
        self.numberofwords = 0
        self.words = []
        self.stemmer = stemmer.PorterStemmer()
        self.vectorizeContext(context)

    def vectorizeContext(self,context):
        [left, _, right] = re.split(constants.HEAD, context)
        left_iter = re.finditer(constants.POS, left)
        right_iter = re.finditer(constants.POS, right)
        F1_l, F2_l = self.removeStopWords(left_iter)
        F1_r, F2_r = self.removeStopWords(right_iter)
        self.getWindow(F1_l, F2_l, right=False)
        self.getWindow(F1_r, F2_r, right=True)
        self.words = Counter(self.words)

    def removeStopWords(self, iterable):
        F1 = []
        F2 = []
        position = 0
        for iter in iterable:
            self.numberofwords += 1
            position += 1
            word = iter.group(1).strip()
            if word not in constants.STOP_WORDS:
                word = self.stemmer.stem(word, 0, len(word) - 1)
                pos = iter.group(2)
                F1.append(word)
                F2.append(pos)
                self.words.append(pos)
            self.words.append(word)

        return F1, F2

    def getWindow(self, F1, F2, right):
        if right:
            bound = 3 if len(F1) >= 3 else len(F1)
            for i in range(bound):
                self.feature.append(F1[i])
                self.feature.append('{pos}{i}'.format(pos=F2[i], i=i+1))
        else:
            bound = len(F1) - 4 if len(F1) >= 3 else -1
            position = -1
            for i in range(len(F1)-1, bound, -1):
                self.feature.append(F1[i])
                self.feature.append('{pos}{i}'.format(pos=F2[i], i=position))
                position -= 1


class Model:

    def __init__(self, trainpos, testpos):
        self.trainpos = trainpos
        self.testpos = testpos
        self.lexelts = []

    def _readFile(self, test=False):
        if test:
            fp = open(self.testpos)
        else:
            fp = open(self.trainpos)
        return fp.read()

    def extractLexelt(self):
        data = self._readFile()
        lexelts = re.finditer(constants.LEXELT, data)
        for lexelt in lexelts:
            self.lexelts.append(Lexelt(lexelt.group(1)))
            self._extractInstance(lexelt.group(2))

    def _extractInstance(self, bulk_instance):
        instances = re.finditer(constants.INSTANCE, bulk_instance)
        for instance in instances:
            self._extractSense(instance.group(2))

    def _extractSense(self, instance):
        answer = next(re.finditer(constants.ANSWER, instance))
        senseid = answer.group(2)
        context = answer.group(3)
        self.lexelts[-1].addSense(Sense(senseid, context))

    def getLexelt(self, lexeltname):
        for lexelt in self.lexelts:
            if lexelt.name == lexeltname:
                return lexelt

    def predict(self):
        fp = open('results.txt', 'w')
        data = self._readFile(test=True)
        lexelts = re.finditer(constants.LEXELT, data)
        for lexlt in lexelts:
            lexelt = self.getLexelt(lexlt.group(1))
            bulk_instance = lexlt.group(2)
            self.predictForLexelt(lexelt, bulk_instance,fp)

    def predictForLexelt(self, lexelt, bulk_instance,fp):
        instances = re.finditer(constants.INSTANCE, bulk_instance)
        for instance in instances:
            context = next(re.finditer(constants.CONTEXT, instance.group(2))).group(1)
            test_sense = Sense(instance.group(1), context)
            fp.write('{id} {sense}\n'.format(id=test_sense.id, sense=self.predictForSense(lexelt, test_sense)))

            del test_sense

    def predictForSense(self, lexelt, sense):
        answers = []
        ids = list(lexelt.sense_ids)
        for senseid in ids:
            ans = lexelt.getPrior(senseid)
            for item in sense.feature:
                ans *= lexelt.getLikelihood(item, senseid)
            answers.append(ans)
        return ids[np.argmax(answers)]




m = Model('trainS1.pos', 'testS1.pos')
m.extractLexelt()
m.predict()

