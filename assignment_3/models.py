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

    def vectorizeContext(self,context):
        [left, _, right] = re.split(constants.HEAD, context)
        left_iter = re.finditer(constants.POS, left)
        right_iter = re.finditer(constants.POS, right)
        F1_l, F2_l = self.removeStopWords(left_iter)
        F1_r, F2_r = self.removeStopWords(right_iter)
        self.getWindow(F1_l, F2_l, right=False)
        self.getWindow(F1_r, F2_r, right=True)
        self.words = Counter(self.words)

    def train(self):
        for context in self.context:


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
        for sense, value in self.senses.items():
            value.train()
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
