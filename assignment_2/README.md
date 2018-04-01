# Assignment 2 : Spelling Correction
Python version 3.5


Run:
```sh
$ python3 assignment2.py dataset.txt output.txt
```
### Details About Algorithm and Code

  - hmm.py : represents hidden markov model and hold the probabilities and vocabulary
  - viterbi.py : do the viterbi calculation to and calculates accuracy
  - constants.py : hold the regex
  - assignment2.py : main script to run to train the model and make calculation

Node classes are representing calculation points in viterbi. They only hold the last word that calculated, probability and corrected word to calculate accuracy later on.
So there is no need to backtrace and follow the words in reverse order to rebuild sentence. As only error points are replaced with the corrected words, nodes are only holds corrected words from the past.
To summarize, nodes are building cumulatively and holds only necessary data to correct sentence and helps calculate accuracy.

