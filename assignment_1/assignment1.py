import utils
import model
import sys

def prepeareModel(fileName):
    mailBodies = utils.extractMailBodies(fileName)
    limit = int(len(mailBodies) * 6 / 10)
    sentences = utils.extractSentences(mailBodies[:limit])
    m = model.Model(sentences=sentences, test_data=mailBodies[limit:])
    m.train()
    return m

if __name__ == "__main__":
    csvFile = sys.argv[1]
    outputFile = sys.argv[2]

    model = prepeareModel(csvFile)
    testResults = model.calculateTestProbs()
    file = open(outputFile, "w")
    file.write("N-gram models are training...\n\n")
    file.write("\n##################################################################################\n")
    file.write("Test Data Smoothed Trigram Probability Results\n\n\n")
    i = 1
    for test in testResults:
        file.write("{mail}. mail log probability : {prob}\n\n".format(mail=i,prob=test))
        i+=1
    file.write("\n##################################################################################\n")
    file.write("Generated Mails From Unigram Model And Their Perplexity And Probabilities\n")
    file.write("-Not Smoothed-\n\n")
    file.write("\n----------------------------------------------------------------------------------\n")
    for i in range(10):
        file.write("{0}.Mail :\n\n".format(i+1))
        mail, perp, prob = model.generateUnigramMail()
        file.write(mail)
        file.write("\n\nPerplexity = {perp} - Probability = {prob}\n\n".format(perp=perp, prob=prob))
        file.write("\n----------------------------------------------------------------------------------\n")

    file.write("\n-Smoothed-\n\n")
    file.write("\n----------------------------------------------------------------------------------\n")
    for i in range(10):
        file.write("{0}.Mail :\n\n".format(i+1))
        mail, perp, prob = model.generateUnigramMail(smooth=True)
        file.write(mail)
        file.write("\n\nPerplexity = {perp} - Probability = {prob}\n\n".format(perp=perp, prob=prob))
        file.write("\n----------------------------------------------------------------------------------\n")

    file.write("\n##################################################################################\n")
    file.write("Generated Mails From Bigram Model And Their Perplexity And Probabilities\n")
    file.write("-Not Smoothed-\n\n")
    file.write("\n----------------------------------------------------------------------------------\n")
    for i in range(10):
        file.write("{0}.Mail :\n\n".format(i+1))
        mail, perp, prob = model.generateBigramMail()
        file.write(mail)
        file.write("\n\nPerplexity = {perp} - Probability = {prob}\n\n".format(perp=perp, prob=prob))
        file.write("\n----------------------------------------------------------------------------------\n")

    file.write("\n-Smoothed-\n\n")
    file.write("\n----------------------------------------------------------------------------------\n")
    for i in range(10):
        file.write("{0}.Mail :\n\n".format(i+1))
        mail, perp, prob = model.generateBigramMail(smooth=True)
        file.write(mail)
        file.write("\n\nPerplexity = {perp} - Probability = {prob}\n\n".format(perp=perp, prob=prob))
        file.write("\n----------------------------------------------------------------------------------\n")

    file.write("\n##################################################################################\n")
    file.write("Generated Mails From Trigram Model And Their Perplexity And Probabilities\n")
    file.write("-Not Smoothed-\n\n")
    file.write("\n----------------------------------------------------------------------------------\n")
    for i in range(10):
        file.write("{0}.Mail :\n\n".format(i+1))
        mail, perp, prob = model.generateTrigramMail()
        file.write(mail)
        file.write("\n\nPerplexity = {perp} - Probability = {prob}\n\n".format(perp=perp, prob=prob))
        file.write("\n----------------------------------------------------------------------------------\n")

    file.write("\n-Smoothed-\n\n")
    file.write("\n----------------------------------------------------------------------------------\n")
    for i in range(10):
        file.write("{0}.Mail :\n\n".format(i+1))
        mail, perp, prob = model.generateTrigramMail(smooth=True)
        file.write(mail)
        file.write("\n\nPerplexity = {perp} - Probability = {prob}\n\n".format(perp=perp, prob=prob))
        file.write("\n----------------------------------------------------------------------------------\n")












