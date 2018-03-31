import constants
import re

# nodes are representing correction points and holds corresponding data
class Node:
    def __init__(self, initialprob=1, word=None):
        self.prob = initialprob
        self.lastword = word
        self.corrections = []

# by using the given hidden markov model makes corrections and calculate accuracy for given data
class Viterbi:
    def __init__(self, dataset, model, output):
        self.dataset = dataset
        self.hmm = model # given hidden markov model
        self.lines = []
        self.output = open(output, 'w')

    # calculates node with the maximum probability
    def _maxNode(self, nodeList):
        max_Node = nodeList[0]
        for node in nodeList:
            if node.prob > max_Node.prob:
                max_Node = node
        return max_Node

    def _readData(self):
        fp = open(self.dataset)
        return fp.readlines()

    # main method for corrections and accuracy calculation
    def calculateAccuracy(self):
        # initial values for true corrections and all corrections
        check_true = 0
        checks = 0
        # every line in data make correction
        for line in self._readData():
            base_line = line
            # initial node : for each sentence start with node with 1 probability
            nodes = [Node()]
            # find the error areas in lines
            errors = re.finditer(constants.ERROR_REGEX, line)
            # boolean to understand if corresponding calculation is done for sentence begin
            # as those calculation needs to add <s> flag at the beginning of first word
            isBegin = True
            # hold right words to calculate accuracy of corrections
            rightversions = []
            for error in errors:
                # for each error seperate the sentence into two
                # left hand side will be bigram calculation
                # right hand side will be handled later correction nodes are created
                [left, right] = line.strip().split(error.group(0), 1)
                line = right

                # if left hand side is not empty calculate bigram and multiply the
                # probability for each node
                if left != '':
                    for node in nodes:
                        prob, node.lastword = self._calculateCommonBigramProb(left, isBegin)
                        node.prob *= prob

                # hold the right version (target)
                rightversions.append(error.group(1).lower())
                mispelled = error.group(2).lower()
                # if model has candidates for corresponding misspelled word
                if self.hmm.inVocab(mispelled):
                    # take the nodes that calculated before as temp
                    temp_nodes = nodes
                    # as nodes will be change after this calculation reset it
                    nodes = []
                    for candidate in self.hmm.misvocab[mispelled]:
                        # hold candidate probabilities for each candidate word of misspelled
                        cand_nodes = []
                        for node in temp_nodes:
                            # if sentence started with error
                            if node.lastword is None:
                                node.lastword = constants.START
                            # calculate the probability of correction
                            # probability from left hand side * emission probability * transission probability
                            prob = node.prob \
                                   * self.hmm.getEditProb(mispelled.lower(), candidate.lower()) \
                                   * self.hmm.getBigramProb(node.lastword.lower(), candidate.lower())
                            cand_nodes.append(Node(prob, candidate))
                            # take the correction list from the parent node
                            cand_nodes[-1].corrections = node.corrections[:]
                            # and add the new one
                            cand_nodes[-1].corrections.append(candidate)
                        # for candidate append the node with a max probability
                        nodes.append(self._maxNode(cand_nodes))
                else:
                    # if misspelled word did not found then act as it is wrong guessed
                    for node in nodes:
                        node.lastword = error.group(2)
                        node.corrections.append(error.group(2))
                # if it comes to this point this means that it pass the beginning
                isBegin = False

            # get the winner node from the final nodes for line
            winner = self._maxNode(nodes).corrections
            # if any correction has done print the result and calculate the accuracy meanwhile
            if len(winner) > 0:
                items = zip(winner, rightversions)
                for cand, right in items:
                    if cand == right:
                        check_true += 1
                    checks += 1
                self.printOut(base_line, winner)
            else :
                self.output.write(base_line + '\n')
        # after finishing all data print out the accuracy
        self.output.write("ACCURACY :")
        self.output.write("%0.2f" % (check_true/checks))

    # print out the corrected line with before after version
    def printOut(self, base_sentence, corrections):
        i=0
        errors = re.finditer(constants.ERROR_REGEX, base_sentence)
        for error in errors:
            base_sentence = base_sentence.replace(error.group(0), corrections[i])
            i += 1

        self.output.write(base_sentence+'\n')


    # calculate the bigram probability for given line part
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



