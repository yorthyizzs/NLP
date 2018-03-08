import pickle
import Utils
import model

"""
fileName = 'emails.csv'
mailBodies = Utils.extractMailBodies(fileName)
sentences = Utils.extractSentences(mailBodies)[:100]
"""



with open('cache/body_list.pickle', 'rb') as handle:
    mailBodies = pickle.load(handle)

with open('cache/sentences.pickle', 'rb') as handle:
    sentences = pickle.load(handle)


m = model.Model(sentences=sentences[:100], test_data=mailBodies[100:110])
m.train()
print(m.generateUnigramMail())
print(m.generateBigramMail())
print(m.generateTrigramMail())
print(m.generateUnigramMail(smooth=True))
print(m.generateBigramMail(smooth=True))
print(m.generateTrigramMail(smooth=True))
print(m.calculateTestProbs())
