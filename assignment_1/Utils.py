import pandas as pd
import re
import pickle
import Unigram,Bigram,Trigram
import numpy
numpy.set_printoptions(threshold=numpy.nan)

base_mail_regex =r'X-FileName:.*[\n]+'
forwarded_mail_regex = r'---------------------- Forwarded(.|\n)*Subject:.*[\n]+'
nested_mail_regex = r'Subject:.*[\n]+'
sentence_regex = r'(?<=\?|!|\.)\s*(?=[A-Z]|$)'
sentence_begin = '<s>'
sentence_end = '<\s>'

def exportNestedMail(mailBody,bodyList):
    nested_body = re.split(forwarded_mail_regex, mailBody)
    for nested in nested_body[:]:
        if 'Subject' in nested:
            new_arr = (re.split(nested_mail_regex,nested))
            appendToBodyList(new_arr[1:], bodyList)
        else:
            appendToBodyList([nested], bodyList)

def appendToBodyList(bodies, bodyList):
    for body in bodies:
        if body != '' and body != '\n' and not ('Subject:' in body) :
            bodyList.append(body)

def extractMailBodies(csvFile, bodyList):
    enronFile = pd.read_csv(csvFile)
    for mail in enronFile['message']:
        basic_bodies = re.split(base_mail_regex, mail)
        for basic in basic_bodies[1:]:
            if 'Subject' in basic:
                exportNestedMail(basic, bodyList)
            else:
                bodyList.append(basic)
    return bodyList

def extractSentences(bodyList):
    sentences = []
    for body in bodyList:
        sentences.extend(re.split(sentence_regex, body))
    return sentences

def extractWords(sentences):
    wordList = []
    wordSet = set()
    for sentence in sentences:
        splitted = (re.split('(\)|\(|/|\?|!|\.|,|;|:|\n|\s)', sentence))
        splitted = [w for w in splitted if (w != '' and w != ' ' and w != '\n')]
        if splitted:
            splitted.insert(0, sentence_begin)
            splitted.append(sentence_end)
            wordList.append(splitted)
            wordSet |= set(splitted)
    return wordList,wordSet

"""
body_list = []
body_list = extractMailBodies('emails.csv', body_list)
with open('body_list.pickle', 'wb') as handle:
    pickle.dump(body_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('body_list.pickle', 'rb') as handle:
    body_list = pickle.load(handle)

sentences = extractSentences(body_list)

with open('sentences.pickle', 'wb') as handle:
    pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('sentences.pickle', 'rb') as handle:
    sentences = pickle.load(handle)

"""
body_list = []
body_list = extractMailBodies('emails.csv', body_list)
sentences = extractSentences(body_list)
wlist, wset = (extractWords(sentences[:100]))
print(len(wset))

unigram = Trigram.Trigram(words=wset, sentences=wlist[:100],smooth=True)
unigram.countTheWords()
unigram.calculateTheProbs()
#unigram.calculatePerplexity()
sent, perp = (unigram.generateMail())
print(sent)
print(perp)
#print(unigram.perplexity)
print(unigram.probs.shape)






