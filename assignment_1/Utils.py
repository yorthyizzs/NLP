import pandas as pd
import re
import constants

def exportNestedMail(mailBody,bodyList):
    nested_body = re.split(constants.forwarded_mail_regex, mailBody)
    for nested in nested_body[:]:
        if 'Subject' in nested:
            new_arr = (re.split(constants.nested_mail_regex,nested))
            appendToBodyList(new_arr[1:], bodyList)
        else:
            appendToBodyList([nested], bodyList)

def appendToBodyList(bodies, bodyList):
    for body in bodies:
        if body != '' and body != '\n' and not ('Subject:' in body) :
            bodyList.append(body)

def extractMailBodies(csvFile):
    enronFile = pd.read_csv(csvFile)
    bodyList = []
    for mail in enronFile['message']:
        basic_bodies = re.split(constants.base_mail_regex, mail)
        for basic in basic_bodies[1:]:
            if 'Subject' in basic:
                exportNestedMail(basic, bodyList)
            else:
                bodyList.append(basic)
    return bodyList

def extractSentences(bodyList):
    sentences = []
    for body in bodyList:
        sentences.extend(re.split(constants.sentence_regex, body))
    return sentences







