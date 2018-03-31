import hmm
import viterbi

if __name__ == '__main__':
    import sys
    input = sys.argv[1]
    output = sys.argv[2]

    model = hmm.HiddenMarkovModel(input)
    model.train()

    vito = viterbi.Viterbi(input, model, output)
    vito.calculateAccuracy()

