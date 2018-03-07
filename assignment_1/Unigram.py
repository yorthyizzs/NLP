import numpy as np
import random

class Unigram :
    def __init__(self, sentences, words, smooth=False):
        self.smooth = smooth
        self.sentences = sentences
        self.words = np.array(list(words))
        self.wordCounts = None
        self.counts = np.zeros(len(words))
        self.probs = np.zeros(len(words))

    def countTheWords(self):
        for sentence in self.sentences:
            for word in sentence[1:len(sentence)-1]:
                # as <s> does not count in unigram start the first word of sentence
                # and also i did not count <\s> because stop conditions are ./!/?
                # and that is the only way to make it work with unigram.
                self.counts[np.argwhere(self.words == word)[0]] += 1

    def calculateTheProbs(self):
        if self.smooth:
            add_one = self.counts+1
            self.probs = add_one / np.sum(add_one)
        else:
            self.probs = self.counts/np.sum(self.counts)

    def generateMail(self):
        wordCount = 0
        word = None
        sentence = ''
        log_probs = np.log2(self.probs) # in the mean time calculate perplexity
        perplexity = 0
        while wordCount<31 and word != '.' and  word != '!' and word !='?': #stop if there is 30 words or stopping punctions has seen
            rand_prob = random.uniform(0, np.max(self.probs))
            #create random probability
            minOfMax = np.min(self.probs[self.probs > rand_prob])
            #find the max range of corresponding word
            wordIndex = np.argwhere(self.probs == minOfMax)[0]
            #find that word's index if there is more than one pick first one
            word = self.words[wordIndex]
            #get the randomly guessed word
            wordCount += 1
            sentence = sentence + word[0] + ' ' #create the sentence
            perplexity += log_probs[wordIndex] #calculate perplexity in the same time
        #when generating process has done finish the perplexity calculation
        perplexity = 2 ** (-1 / wordCount * perplexity)
        return sentence, perplexity[0]



