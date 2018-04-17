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
        self.items = {}
        
    def getPriorDivident(self):
        return sum(self.items.values())

    def getLikelihood(self, item):
        if item in self.items.keys():
            return float(self.items[item]/sum(self.items.values()))
        else:
            return -1

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
            F1.append(iter.group(1).strip())
            F2.append(iter.group(2))
        return F1, F2

    def train(self):
        for context in self.context:
            self.vectorizeContext(context)
        self.items = Counter(self.feature)

    def getPredictionProb(self, features, N):
        ans = float(self.getPriorDivident()/N)
        for word in features:
            ans *= self.getLikelihood(word)
        return ans


class Lexelt:
    def __init__(self, name, instances):
        self.name = name
        self.instances = instances
        self.senses = {}
        self.totalItems = 0
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
            self.totalItems += sum(sense.items.values())

    def test(self, instance_bulk, N, fp):
        instances = re.finditer(constants.INSTANCE, instance_bulk)
        for instance in instances:
            context = next(re.finditer(constants.CONTEXT, instance.group(2))).group(1)
            temp_sense = Sense(instance.group(1), context)
            temp_sense.train()
            fp.write('{id} {sense}\n'.format(id=temp_sense.id, sense=self.predict(temp_sense, N)))

    def maxId(self, probs, senseids):
        i = 0
        for j in range(len(probs)):
            if probs[j] > probs[i]:
                i = j
        return senseids[i]

    def predict(self, test, N):
        probs = []
        senseids = []
        for senseid, sense in self.senses.items():
            senseids.append(senseid)
            probs.append(sense.getPredictionProb(test.feature, N))
        return self.maxId(probs, senseids)


class NaiveBayesClassifier:
    def __init__(self, trainpos, testpos, output):
        self.trainpos = trainpos
        self.testpos = testpos
        self.lexelts = []
        self.N = 0
        self.output = open(output, 'w')
        self._train()
        self._test()

    def _getLexelt(self, lexeltname):
        for lexelt in self.lexelts:
            if lexelt.name == lexeltname:
                return lexelt

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
            self.N += self.lexelts[-1].totalItems

    def _train(self):
        self._extractLexelt()

    def _test(self):
        data = self._readFile(test=True)
        lexelts = re.finditer(constants.LEXELT, data)
        for lex_match in lexelts:
            lexelt = self._getLexelt(lex_match.group(1))
            instances = lex_match.group(2)
            lexelt.test(instances, self.N, self.output)



if __name__ == '__main__':
    import sys
    NaiveBayesClassifier(sys.argv[1], sys.argv[2], sys.argv[3])
