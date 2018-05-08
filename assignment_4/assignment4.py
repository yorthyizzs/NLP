import tensorflow as tf
import data_handling as dh

class Model:
    def __init__(self, pos, neg, vec,percentage, lr=0.01, training_epochs=30, yorthy=False):
        self.lr = lr
        self.training_epochs = training_epochs
        self.n_input = 200
        self.n_classes = 2
        self.weights = {}
        self.biases = {}
        # input and output layers
        self.input = tf.placeholder("float", [None, self.n_input])
        self.output = tf.placeholder("float", [None, self.n_classes])
        # get the model architecture (you have two option one is from assignment second is made by me :) )
        self.network = self.initialize_model(yorthy)

        # loss optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.network, labels=self.output))
        #train optimizer (Gradient Descent)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss_op)
        # defitiniton of correctness (accuracy formula)
        self.correct_pred = tf.equal(tf.argmax(tf.nn.softmax(self.network), 1), tf.argmax(tf.nn.softmax(self.output), 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        # initial values
        self.init = tf.global_variables_initializer()
        # get the data in splitted and shuffled format
        self.train_feats, self.test_feats, self.train_labels, self.test_labels = self.train_test_split(pos, neg, vec, percentage)
        #train the model and show the accuracy results
        self.train()

    def train_test_split(self, pos, neg, vec, percentage):
        # data come here in shuffled way
        features, labels = dh.createInputs(pos, neg, vec)
        # calculate the wanted percentage
        data_size = labels.shape[0]
        train_size = int(data_size*percentage/100)
        return features[:train_size, :], features[train_size:, :], labels[:train_size, :], labels[train_size:, :]

    def initialize_model(self, yorthy = None):
        if yorthy:
            n_hidden_1 = 100  # 1st layer number of features
            n_hidden_2 = 50  # 2nd layer number of features
            n_hidden_3 = 20 # 3rd layer number of features
            # Store layers weight & bias
            self.weights = {
                'h1': tf.Variable(tf.random_normal([self.n_input, n_hidden_1])),
                'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
                'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
                'out': tf.Variable(tf.random_normal([n_hidden_3, self.n_classes]))
            }
            self.biases = {
                'b1': tf.Variable(tf.random_normal([n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([n_hidden_2])),
                'b3': tf.Variable(tf.random_normal([n_hidden_3])),
                'out': tf.Variable(tf.random_normal([self.n_classes]))
            }
            return self.yorthy_model()
        else:
            n_hidden_1 = 100  # 1st layer number of features
            n_hidden_2 = 100  # 2nd layer number of features
            # Store layers weight & bias
            self.weights = {
                'h1': tf.Variable(tf.random_normal([self.n_input, n_hidden_1])),
                'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
                'out': tf.Variable(tf.random_normal([n_hidden_2, self.n_classes]))
            }
            self.biases = {
                'b1': tf.Variable(tf.random_normal([n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([n_hidden_2])),
                'out': tf.Variable(tf.random_normal([self.n_classes]))
            }
            return self.base_model()

    def base_model(self):
        layer_1 = tf.add(tf.matmul(self.input, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        return tf.matmul(layer_2, self.weights['out']) + self.biases['out']

    def yorthy_model(self):
        # this model made by me
        # there is no logical reason behind the layers
        # I only did some experiments and I left when I get better accurucies than the other model
        # I wanted to make some readings but I could not find a time for it
        # So I used my experiments results to build a model. :)
        layer_1 = tf.add(tf.matmul(self.input, self.weights['h1']), self.biases['b1'])
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_3 = tf.add(tf.matmul(layer_2, self.weights['h3']), self.biases['b3'])
        layer_3 = tf.nn.selu(layer_3)
        return tf.matmul(layer_3, self.weights['out']) + self.biases['out']

    def train(self):
        with tf.Session() as sess:
            # run the initial
            sess.run(self.init)
            # run the epochs
            for i in range(self.training_epochs):
                sess.run(self.train_op, feed_dict={self.input: self.train_feats, self.output: self.train_labels})
                loss, acc = sess.run([self.loss_op, self.accuracy], feed_dict={
                    self.input: self.train_feats, self.output: self.train_labels
                })
                print("For Epoch = {}/{} Training accuracy = {},  Training loss = {}".format(i+1, self.training_epochs, acc, loss))

            # use the trained model on test data
            print("RESULT = Testing Accuracy:", self.accuracy.eval({
                self.input: self.test_feats, self.output: self.test_labels
            }))

if __name__ == '__main__':
    import sys

    pos = sys.argv[1]
    neg = sys.argv[2]
    vec = sys.argv[3]
    percentage = int(sys.argv[4])
    if len(sys.argv) > 5:
        if sys.argv[5] == 'yorthy':
            Model(pos, neg, vec, percentage, yorthy=True)
        else:
            print("There is no such a model !")
    else:
        Model(pos, neg, vec, percentage)
