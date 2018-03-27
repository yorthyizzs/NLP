import re
import constants


def correctLine(line):
    errors = re.finditer(constants.ERROR_REGEX, line)
    for error in errors:
        line = line.replace(error.group(0), error.group(1))
    return line

def readData(file):
    fp = open(file)
    return fp.readlines()

def correctData(lines):
    for line in lines:
        words = (re.split(constants.WORDS_EXTRACTER, correctLine(line)))
        print([w for w in  words if (w != '' and w != ' ' and w != '\n') ])




lines = readData('dataset.txt')
correctData(lines)
