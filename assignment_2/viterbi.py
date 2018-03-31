import constants
import re
import hmm

class Node:
    def __init__(self, initialprob=1, word=None):
        self.prob = initialprob
        self.lastword = word
        self.corrections = []


class Viterbi:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.hmm = model
        self.lines = []

    def _maxNode(self, nodeList):
        max_Node = nodeList[0]
        for node in nodeList:
            if node.prob > max_Node.prob:
                max_Node = node
        return max_Node

    def _readData(self):
        fp = open(self.dataset)
        return fp.readlines()

    def calculateAccuracy(self):
        check_true = 0
        checks = 0
        for line in self._readData():
            line1 = line
            nodes = [Node()]
            errors = re.finditer(constants.ERROR_REGEX, line)
            isBegin = True
            rightversions = []
            for error in errors:
                [left, right] = line.strip().split(error.group(0), 1)
                line = right

                if left != '':
                    for node in nodes:
                        prob, node.lastword = self._calculateCommonBigramProb(left, isBegin)
                        node.prob *= prob

                rightversions.append(error.group(1).lower())
                mispelled = error.group(2).lower()
                if self.hmm.inVocab(mispelled):
                    ##TO DO : if not in vocab break it 
                    temp_nodes = nodes
                    nodes = []
                    for candidate in self.hmm.misvocab[mispelled]:
                        cand_nodes = []
                        for node in temp_nodes:
                            if node.lastword is None :
                                node.lastword = constants.START
                            prob = node.prob \
                                   * self.hmm.getEditProb(mispelled.lower(), candidate.lower()) \
                                   * self.hmm.getBigramProb(node.lastword.lower(), candidate.lower())
                            cand_nodes.append(Node(prob, candidate))
                            cand_nodes[-1].corrections = node.corrections[:]
                            cand_nodes[-1].corrections.append(candidate)

                        nodes.append(self._maxNode(cand_nodes))
                    isBegin = False
                else:
                    break
            if len(nodes) > 0:
                winner = self._maxNode(nodes).corrections
                items = zip(winner, rightversions)
                for cand, right in items:
                    if cand == right:
                        check_true += 1
                    checks +=1

            """        
            
            print(line1)
            print(rightversions)
            print(self._maxNode(nodes).corrections)
            print("----------------------------------------------")"""
        print(check_true/ checks)


    def _calculateCommonBigramProb(self, line, isBegin):
        words = self._extractWord(line)
        prob = 1
        if len(words) != 0:
            if isBegin:
                words.insert(0, constants.START)
            for i in range(1, len(words)):
                prob *= self.hmm.getBigramProb(words[i-1], words[i])
            return prob, words[-1]
        else:
            return 1, None


    def _extractWord(self,line):
        line = re.split(constants.WORDS_EXTRACTER, line)  # extract the words and punctuations
        return [w.lower() for w in line if(w != '' and w != ' ' and
                                                 w != '\n' and w != '"' and
                                                 w != '.' and w != ',' and
                                                 w != '!' and w != '?' and
                                                 w != ';'and w != '/' and
                                                 w != '\\')]



