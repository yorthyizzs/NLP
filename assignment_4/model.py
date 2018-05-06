import tensorflow as tf
import data_handling as dh

class Model:
    def __init__(self, percentage, lr=0.001, training_epochs=20, yorthy=None):
        self.lr = lr
        self.training_epochs = training_epochs
        self.n_input = 200
        self.n_classes = 2
        self.weights = {}
        self.biases = {}
        # tf Graph input
        self.input = tf.placeholder("float", [None, self.n_input])
        self.output = tf.placeholder("float", [None, self.n_classes])
        self.network = self.initialize_model(yorthy)

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.network, labels=self.output))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss_op)

        self.correct_pred = tf.equal(tf.argmax(tf.nn.softmax(self.network), 1), tf.argmax(tf.nn.softmax(self.output), 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.init = tf.global_variables_initializer()
        self.train_feats, self.test_feats, self.train_labels, self.test_labels = self.train_test_split(percentage)
        self.train()

    def train_test_split(self, percentage):
        features, labels = dh.createInputs('positives.txt', 'negatives.txt', 'vectors.txt')
        data_size = labels.shape[0]
        train_size = int(data_size*percentage/100)
        return features[:train_size, :], features[train_size:, :], labels[:train_size, :], labels[train_size:, :]

    def initialize_model(self, yorthy = None):
        if yorthy:
            print("yorthy model here")
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
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(self.input, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        return tf.matmul(layer_2, self.weights['out']) + self.biases['out']


    def train(self):
        with tf.Session() as sess:
            # Run the initializer
            sess.run(self.init)
            for i in range(self.training_epochs):
                sess.run(self.train_op, feed_dict={self.input: self.train_feats, self.output: self.train_labels})
                loss, acc = sess.run([self.loss_op, self.accuracy], feed_dict={
                    self.input: self.train_feats, self.output: self.train_labels
                })
                print("epoch= {} acc = {}, loss = {}".format(i, acc, loss))

            print("Testing Accuracy:", self.accuracy.eval({
                self.input: self.test_feats, self.output: self.test_labels
            }))

m = Model(75)
