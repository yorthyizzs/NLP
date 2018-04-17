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
            # if that word never seen before handle the likelihood in the caller method
            return -1

    def vectorizeContext(self,context):
        # collect the corresponding feature by using context of the sense
        # split the context according to head and take the left and right hand side
        [left, _, right] = re.split(constants.HEAD, context)
        # extract the items of each side as F1 and F2 vectors
        lf1, lf2 = self.extractFeature(re.finditer(constants.POS, left))
        rf1, rf2 = self.extractFeature(re.finditer(constants.POS, right))
        # get the window of those vectors (window size = 3)
        self.getWindow(lf1, lf2, False)
        self.getWindow(rf1, rf2, True)

    def getWindow(self, F1, F2, right):
        # collect the features in window and name according to that
        # ex: if right -> pos1 else pos-1
        if right:
            bound = 3 if len(F1) > 3 else len(F1)
            for i in range(bound):
                if F1[i] not in constants.STOP_WORDS:
                    self.feature.append(self.stemmer.stem(F1[i], 0, len(F1[i])-1))
                    self.feature.append('{pos}{i}'.format(pos=F2[i], i=i+1))
        else:
            bound = len(F1) - 4 if len(F1) > 3 else -1
            position = -1
            for i in range(len(F1)-1, bound, -1):
                if F1[i] not in constants.STOP_WORDS:
                    self.feature.append(self.stemmer.stem(F1[i], 0, len(F1[i])-1))
                    self.feature.append('{pos}{i}'.format(pos=F2[i], i=position))
                position -= 1

    def extractFeature(self, iterable):
        # extract the words and the pos tags in context
        # create vectors according to that
        F1 = [] # words
        F2 = [] # pos tags
        for iter in iterable:
            F1.append(iter.group(1).strip())
            F2.append(iter.group(2))
        return F1, F2

    def train(self):
        # train the sense by using it's contexts
        for context in self.context:
            self.vectorizeContext(context)
        # after train phase and count the features
        # that is collected in window size
        self.items = Counter(self.feature)

    def getPredictionProb(self, features, N):
        # calculate the probability of feature belongs to this sense
        ans = float(self.getPriorDivident()/N)
        for word in features:
            likelihood = self.getLikelihood(word)
            # if word does not seen before assign the 1/N as a likelihood
            ans *= likelihood if likelihood != -1 else float(1/N)
        return ans


class Lexelt:
    def __init__(self, name, instances):
        self.name = name
        self.instances = instances
        self.senses = {}
        self.totalItems = 0
        self._train()

    def _extractInstance(self):
        # extract each sense that belongs to lexelt
        instances = re.finditer(constants.INSTANCE, self.instances)
        for instance in instances:
            self._extractSense(instance.group(2))

    def _extractSense(self, instance):
        # collect the senses by creating sense object
        answer = next(re.finditer(constants.ANSWER, instance))
        senseid = answer.group(2)
        context = answer.group(3)
        # if sense id seen before just add the context to that sense
        if senseid in self.senses.keys():
            self.senses[senseid].context.append(context)
        # if sense id never seen before create the sense
        else:
            self.senses[senseid] = Sense(senseid, context)

    def _train(self):
        # extract each instance
        self._extractInstance()
        # after initializing senses train each sense that belongs to this lexelt
        for sense_id, sense in self.senses.items():
            sense.train()
            # count the collected items
            self.totalItems += sum(sense.items.values())

    def test(self, instance_bulk, N, fp):
        # test each context in this lexelt
        instances = re.finditer(constants.INSTANCE, instance_bulk)
        for instance in instances:
            context = next(re.finditer(constants.CONTEXT, instance.group(2))).group(1)
            # create temporary sense just in case the extract corresponding feature
            temp_sense = Sense(instance.group(1), context)
            temp_sense.train()
            fp.write('{id} {sense}\n'.format(id=temp_sense.id, sense=self.predict(temp_sense, N)))
            # delete the predicted sense
            del temp_sense

    def maxId(self, probs, senseids):
        # primitive version of argmax
        i = 0
        for j in range(len(probs)):
            if probs[j] > probs[i]:
                i = j
        return senseids[i]

    def predict(self, test, N):
        # prediction for test context
        probs = []
        senseids = []
        # for each sense of this lexelt calculate probability
        for senseid, sense in self.senses.items():
            senseids.append(senseid)
            probs.append(sense.getPredictionProb(test.feature, N))
        # return the max one
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
        # return the lexelt object that has a corresponding name
        for lexelt in self.lexelts:
            if lexelt.name == lexeltname:
                return lexelt

    def _readFile(self, test=False):
        if test:
            fp = open(self.testpos)
        else:
            fp = open(self.trainpos)
        return fp.read()

    def _train(self):
        data = self._readFile()
        # extract the lexelts
        lexelts = re.finditer(constants.LEXELT, data)
        # for each lexelt create their objects
        for lexelt in lexelts:
            self.lexelts.append(Lexelt(lexelt.group(1), lexelt.group(2)))
            # every lexelt training itself in initializing phase
            # so we can get the totalitems directly to calculate
            # total feature in each sense in each lexelt
            self.N += self.lexelts[-1].totalItems

    def _test(self):
        data = self._readFile(test=True)
        # extract the lexelts of test data
        lexelts = re.finditer(constants.LEXELT, data)
        for lex_match in lexelts:
            # get the corresponding lexelt
            lexelt = self._getLexelt(lex_match.group(1))
            instances = lex_match.group(2)
            # test each context of lexelt
            lexelt.test(instances, self.N, self.output)


if __name__ == '__main__':
    import sys
    """
    sys.argv[1]: train.pos
    sys.argv[2]: test.pos
    sys.argv[3]: output.txt
    """
    NaiveBayesClassifier(sys.argv[1], sys.argv[2], sys.argv[3])
