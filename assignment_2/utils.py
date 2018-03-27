class Node:
    def __init__(self, word, sentence, prob):
        self.sentence = sentence
        self.prob = prob
        self.lastWord = word
