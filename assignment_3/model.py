import re
import constants
import stemmer
from collections import Counter

class Lexelt:
    def __init__(self, name):
        self.name = name
        self.senses = []
        self.itemCounts = []


    def addSense(self, sense):
        self.senses.append(sense)
        for item in sense.feature:
            self.itemCounts.append(item)

    def calculateFrequencies(self):
        self.itemCounts = Counter(self.itemCounts)

class Sense:
    def __init__(self, id, context):
        self.id = id
        self.feature = []
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

    def removeStopWords(self, iterable):
        F1 = []
        F2 = []
        position = 0
        for iter in iterable:
            position += 1
            word = iter.group(1).strip()
            word = self.stemmer.stem(word, 0, len(word) - 1)
            if word not in constants.STOP_WORDS:
                F1.append(word)
                F2.append(iter.group(2))
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

    def _readFile(self):
        fp = open(self.trainpos)
        return fp.read()

    def extractLexelt(self):
        data = self._readFile()
        lexelts = re.finditer(constants.LEXELT, data)
        for lexelt in lexelts:
            self.lexelts.append(Lexelt(lexelt.group(1)))
            self._extractInstance(lexelt.group(2))
            print(self.lexelts[-1].name)
            self.lexelts[-1].calculateFrequencies()
            print(self.lexelts[-1].itemCounts)
            print("--------------------------------------------------")

    def _extractInstance(self, bulk_instance):
        instances = re.finditer(constants.INSTANCE, bulk_instance)
        for instance in instances:
            self._extractSense(instance.group(2))

    def _extractSense(self, instance):
        answer = next(re.finditer(constants.ANSWER, instance))
        senseid = answer.group(2)
        context = answer.group(3)
        self.lexelts[-1].addSense(Sense(senseid, context))







m = Model('trainS1.pos', 'testS1.pos')
m.extractLexelt()

