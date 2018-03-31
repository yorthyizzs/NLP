import hmm
import viterbi

model = hmm.HiddenMarkovModel('dataset.txt')
model.train()

vit = viterbi.Viterbi('dataset.txt', model)
vit.calculateAccuracy()

