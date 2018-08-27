import tensorflow as tf

class Seq2Seq:

    # 3개의 layer와 128개의 hidden_node 사용
    def __init__(self, vocab_size, n_hidden=128, n_layers=3):
        self.learning_late = 0.001

        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        # one_hot이기 때문에 [batch_size, words_len, vocab_size]
        self.enc_input = tf.placeholder(tf.float32, [None, None, self.vocab_size])
        self.dec_input = tf.placeholder(tf.float32, [None, None, self.vocab_size])
        self.targets = tf.placeholder(tf.int64, [None, None])   # [batct_size, words_len]

        # 학습 총 회수
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        # 모델 구축
        self.build_model()

        # 학습회수 저장
        self.saver = tf.train.Saver(tf.global_variables())

    def build_model(self):
        # cell 구축
        enc_cell, dec_cell = self.build_cells()

        # encode, decode 구축
        with tf.variable_scope('encode'):
            outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, self.enc_input, dtype=tf.float32)

        with tf.variable_scope('decode'):
            outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, self.dec_input, dtype=tf.float32,
                                                    initial_state=enc_states)

        # 학습 모델 구축
        self.logits, self.cost, self.train_op = self.build_ops(outputs, self.targets)

        self.outputs = tf.argmax(self.logits, 2)


    def cell(self, output_keep_prob):
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=output_keep_prob)
        return rnn_cell

    def build_cells(self, output_keep_prob=0.5):
        enc_cell = tf.nn.rnn_cell.MultiRNNCell([self.cell(output_keep_prob)
                                                for _ in range(self.n_layers)])
        dec_cell = tf.nn.rnn_cell.MultiRNNCell([self.cell(output_keep_prob)
                                                for _ in range(self.n_layers)])

        return enc_cell, dec_cell

    def build_ops(self, outputs, targets):

        logits = tf.layers.dense(outputs, self.vocab_size, activation=None)

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets))
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_late).minimize(cost, global_step=self.global_step)

        return logits, cost, train_op

    def train(self, session, enc_input, dec_input, targets):
        return session.run([self.train_op, self.cost],
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input,
                                      self.targets: targets})

    def test(self, session, enc_input, dec_input, targets):
        prediction_check = tf.equal(self.outputs, self.targets)
        accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

        return session.run([self.targets, self.outputs, accuracy],
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input,
                                      self.targets: targets})

    def predict(self, session, enc_input, dec_input):
        return session.run(self.outputs,
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input})