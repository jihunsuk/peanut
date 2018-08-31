import tensorflow as tf

class Hred:

    logits = None
    outputs = None
    cost = None
    train_op = None

    # 3개의 layer와 128개의 hidden_node 사용
    def __init__(self, vocab_size, training_mode, iterNumber,n_hidden=128, n_layers=3):

        self.learning_late = 0.001
        self.vocab_size = vocab_size
        self.embedding_size = 10
        self.init_width = 1 / self.embedding_size
        self.word_embedding_matrix = tf.Variable(
            tf.random_uniform([self.vocab_size, self.embedding_size], -self.init_width, self.init_width, dtype=tf.float32),
            name="embeddings",
            dtype=tf.float32)
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.training_mode = training_mode
        self.iterNumber = iterNumber

        self.enc_input_idx = tf.placeholder(tf.int32, [None, None, 25], name='enc_input_idx')
        self.dec_input_idx = tf.placeholder(tf.int32, [None, None, 25], name='dec_input_idx')
        self.targets = tf.placeholder(tf.int32, [None, None, 25], name='targets')
        self.enc_length = tf.placeholder(tf.int32, [None, None], name='enc_length')
        self.dec_length = tf.placeholder(tf.int32, [None, None], name='dec_length')
        self.dec_max_length = tf.placeholder(tf.int32, [None])
        self.context_size = tf.placeholder(tf.int32, name='context_size')

        # 학습 총 회수
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        # 모델 구축
        self.build_model()

        # 모델저장 객체 생성
        self.saver = tf.train.Saver(tf.global_variables())

    def build_model(self):
        # cell 구축
        enc_cell, dec_cell, context_cell = self.build_cells()

        total_enc_states = []

        for i in range(self.iterNumber):
            with tf.variable_scope('encode', reuse=tf.AUTO_REUSE):
                input_sentence_emb = tf.nn.embedding_lookup(self.word_embedding_matrix, self.enc_input_idx[i])
                outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, input_sentence_emb, self.enc_length[i], dtype=tf.float32)
                total_enc_states.append(enc_states)

        enc_states = tf.convert_to_tensor(total_enc_states)

        with tf.variable_scope('context', reuse=tf.AUTO_REUSE):
            outputs, context_states = tf.nn.dynamic_rnn(context_cell, enc_states, dtype=tf.float32)

        total_outputs = []
        for i in range(self.iterNumber):
            with tf.variable_scope('decode', reuse=tf.AUTO_REUSE):
                if self.training_mode is True:
                    output_sentence_emb = tf.nn.embedding_lookup(self.word_embedding_matrix, self.dec_input_idx[i])
                    helper = tf.contrib.seq2seq.TrainingHelper(output_sentence_emb, self.dec_length[i])
                else:
                    start_tokens = tf.fill([self.context_size], 1) # data를 10개씩 읽어서 context_size는 9
                    end_token = 2
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.word_embedding_matrix, start_tokens, end_token)

                initial_state = tf.reshape(outputs[i], [-1, self.n_hidden])
                initial_state = tf.cast(initial_state, dtype=tf.float32)
                decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, initial_state,
                                                          output_layer=tf.layers.Dense(self.vocab_size))
                output, states, length = tf.contrib.seq2seq.dynamic_decode(decoder)
                total_outputs.append(output)

        self.logits, self.cost, self.train_op = self.build_ops(total_outputs, self.targets)
        self.outputs = tf.identity(total_outputs[0].sample_id, name='outputs')

        prediction_check = tf.equal(self.outputs, self.targets)
        self.accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32), name='accuracy')

    def cell(self, output_keep_prob):
        rnn_cell = tf.nn.rnn_cell.GRUCell(self.n_hidden)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=output_keep_prob)
        return rnn_cell

    def build_cells(self, output_keep_prob=0.5):
        enc_cell = self.cell(output_keep_prob)
        dec_cell = self.cell(output_keep_prob)
        context_cell = self.cell(output_keep_prob)
        return enc_cell, dec_cell, context_cell

    def build_ops(self, outputs, targets):
        with tf.variable_scope('ops', reuse=tf.AUTO_REUSE):
            loss = []
            for i in range(self.iterNumber):
                logits = outputs[i].rnn_output
                f = tf.fill([9, tf.subtract(25, self.dec_max_length[i]), 164], 0.0)
                new_logits = tf.concat([logits, f], axis=1)

                max_time = self.get_max_time(targets[i])
                weights = tf.sequence_mask(self.dec_length[i], max_time, dtype=logits.dtype)
                loss.append(tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(new_logits, targets[i], weights), name='cost'))

        cost = tf.reduce_mean(loss)

        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_late, name="train_op").minimize(cost, global_step=self.global_step)
        return logits, cost, train_op

    def get_max_time(self, tensor):
        return tensor.shape[1].value

    def predict(self, session, enc_input, enc_length, dec_input, dec_length, dec_max_length, context_size):
        return session.run(self.outputs,
                           feed_dict={self.enc_input_idx: enc_input,
                                      self.dec_input_idx: dec_input,
                                      self.enc_length: enc_length,
                                      self.dec_length: dec_length,
                                      self.dec_max_length : dec_max_length,
                                      self.context_size: context_size
                                      })