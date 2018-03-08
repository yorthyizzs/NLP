import pandas as pd
import re
import constants

def exportNestedMail(mailBody,bodyList):
    # this method split forwarded mails and check if there is still sub-mail
    # if so it send it to corresponding method else it simply add it to mail body list
    nested_body = re.split(constants.forwarded_mail_regex, mailBody)
    for nested in nested_body[:]:
        if 'Subject' in nested:
            new_arr = (re.split(constants.nested_mail_regex,nested))
            appendToBodyList(new_arr[1:], bodyList)
        else:
            appendToBodyList([nested], bodyList)

def appendToBodyList(bodies, bodyList):
    # checks if body is approved to be added in the list then add
    for body in bodies:
        if body != '' and body != '\n' and not ('Subject:' in body) :
            bodyList.append(body)

def extractMailBodies(csvFile):
    # extract the most basic mail bodies and check if body has nested mails
    # if so it send mail to corresponding method else it adds to bodylist
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
    # extract the sentences in mail bodies and gather them in one list
    sentences = []
    for body in bodyList:
        sentences.extend(re.split(constants.sentence_regex, body))
    return sentences







