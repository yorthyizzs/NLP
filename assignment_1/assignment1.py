import pickle
import Utils
import model

"""fileName = 'emails.csv'
mailBodies = Utils.extractMailBodies(fileName)
sentences = Utils.extractSentences(mailBodies)[:100]"""

with open('sentences.pickle', 'rb') as handle:
    sentences = pickle.load(handle)
print(len(sentences))
m = model.Model(sentences=sentences[:100])
m.train()
print(m.generateUnigramMail(smooth=True))
print(m.generateBigramMail(smooth=True))
print(m.generateTrigramMail(smooth=True))
