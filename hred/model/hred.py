import numpy as np
import tensorflow as tf   

class hred:

	def __init__(self, voc_size, learning_rate, num_units):
		self.enc_inputs = tf.placeholder(tf.int32, shape=[None, None], name='enc_inputs')
		self.dec_inputs = tf.placeholder(tf.int32, shape=[None, None], name='dec_inputs')
		self.targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')
		self.enc_seq_length = tf.placeholder(tf.int32, shape=[None], name='enc_seq_length')
		self.dec_seq_length = tf.placeholder(tf.int32, shape=[None], name='dec_seq_length')
		self.max_length = tf.placeholder(tf.int32, shape=[], name='max_length')
		self.voc_size = voc_size
		self.learning_rate = learning_rate
		self.num_units = num_units

		self.build()
		self.saver = tf.train.Saver(tf.global_variables())

	def build(self):
		enc_one_hot = tf.one_hot(self.enc_inputs, self.voc_size, name='enc_one_hot')
		dec_one_hot = tf.one_hot(self.dec_inputs, self.voc_size, name='dec_one_hot')

		with tf.variable_scope('encode'):
			enc_cell = tf.nn.rnn_cell.GRUCell(self.num_units)
			_, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_one_hot, self.enc_seq_length, dtype=tf.float32)

			context_input = tf.reshape(enc_states, shape=[-1,1,self.num_units])

		with tf.variable_scope('context'):
			context_cell = tf.nn.rnn_cell.GRUCell(self.num_units)
			context_output, _ = tf.nn.dynamic_rnn(context_cell, context_input, dtype=tf.float32)

			state = tf.reshape(context_output, shape=[-1, self.num_units])

		with tf.variable_scope('decode'):
			dec_cell = tf.nn.rnn_cell.GRUCell(self.num_units)
			helper = tf.contrib.seq2seq.TrainingHelper(dec_one_hot, self.dec_seq_length)
			projection_layer = tf.layers.Dense(self.voc_size)
			decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, state, output_layer=projection_layer)

		outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
		logits = outputs.rnn_output

		self.weights = tf.to_float(tf.sequence_mask(self.dec_seq_length, self.max_length))
		self.loss = tf.contrib.seq2seq.sequence_loss(logits, self.targets, self.weights)
		self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
	
	def train(self, sess, fdtr):
		return sess.run(self.loss, feed_dict=fdtr)