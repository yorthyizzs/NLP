import re
import constants
import stemmer
from collections import Counter

class Sense:
    def __init__(self, id, context):
        self.id = id
        self.words = []
        self.context = [context]
        self.stemmer = stemmer.PorterStemmer()
        self.feature = []
        self.numberofwords = 0
        

    def vectorizeContext(self,context):
        [left, _, right] = re.split(constants.HEAD, context)
        lf1, lf2 = self.extractFeature(re.finditer(constants.POS, left))
        rf1, rf2 = self.extractFeature(re.finditer(constants.POS, right))
        self.getWindow(lf1, lf2, False)
        self.getWindow(rf1, rf2, True)


    def getWindow(self, F1, F2, right):
        if right:
            bound = 3 if len(F1) >= 3 else len(F1)
            for i in range(bound):
                if F1[i] not in constants.STOP_WORDS:
                    self.feature.append(F1[i])
                    self.feature.append('{pos}{i}'.format(pos=F2[i], i=i+1))
        else:
            bound = len(F1) - 4 if len(F1) >= 3 else -1
            position = -1
            for i in range(len(F1)-1, bound, -1):
                if F1[i] not in constants.STOP_WORDS:
                    self.feature.append(F1[i])
                    self.feature.append('{pos}{i}'.format(pos=F2[i], i=position))
                position -= 1

    def extractFeature(self, iterable):
        F1 = []
        F2 = []
        for iter in iterable:
            self.numberofwords += 1
            F1.append(iter.group(1).strip())
            F2.append(iter.group(2))
        return F1, F2

    def train(self):
        for context in self.context:
            self.vectorizeContext(context)

class Lexelt:
    def __init__(self, name, instances):
        self.name = name
        self.instances = instances
        self.senses = {}
        self._train()

    def _extractInstance(self):
        instances = re.finditer(constants.INSTANCE, self.instances)
        for instance in instances:
            self._extractSense(instance.group(2))

    def _extractSense(self, instance):
        answer = next(re.finditer(constants.ANSWER, instance))
        senseid = answer.group(2)
        context = answer.group(3)
        if senseid in self.senses.keys():
            self.senses[senseid].context.append(context)
        else:
            self.senses[senseid] = Sense(senseid, context)

    def _train(self):
        self._extractInstance()
        for sense_id, sense in self.senses.items():
            sense.train()

            
class Model:

    def __init__(self, trainpos, testpos):
        self.trainpos = trainpos
        self.testpos = testpos
        self.lexelts = []
        self._train()

    def _readFile(self, test=False):
        if test:
            fp = open(self.testpos)
        else:
            fp = open(self.trainpos)
        return fp.read()

    def _extractLexelt(self):
        data = self._readFile()
        lexelts = re.finditer(constants.LEXELT, data)
        for lexelt in lexelts:
            self.lexelts.append(Lexelt(lexelt.group(1), lexelt.group(2)))

    def _train(self):
        self._extractLexelt()

m = Model('trainS1.pos', 'testS1.pos')
