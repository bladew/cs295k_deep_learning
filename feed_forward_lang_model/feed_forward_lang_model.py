import tensorflow as tf
import math


class InputData(object):
	def __init__(self, train_path, test_path):
		super(InputData, self).__init__()
		self.words_ids = {}
		self.batch_source = []
		self.test_source = []
		
		word_id = 0
		with open(train_path, 'r') as sentes:
			for line in sentes:
				words = line.split(' ')
				for i in xrange(len(words) - 1):
					if words[i] not in self.words_ids:
						self.words_ids[words[i]] = word_id
						word_id += 1
					self.batch_source.append(self.words_ids[words[i]])

		with open(test_path, 'r') as sentes:
			for line in sentes:
				words = line.split(' ')
				for i in xrange(len(words) - 1):
					self.test_source.append(self.words_ids[words[i]])


	def next_batch(self, size):
		i = 0
		while i + 1 < len(self.batch_source):
			end = min(len(self.batch_source), i + size)
			Xs, Ys = [], [] 
			while i < end and i + 1 < len(self.batch_source):
				Xs.append(self.batch_source[i])
				Ys.append(self.batch_source[i + 1])
				i += 1
			yield Xs, Ys


	def test_data(self):
		return self.test_source[:len(self.test_source) - 1], self.test_source[1:]


	def get_vocab_size(self):
		return len(self.words_ids)


class BigramInNN(object):
	def __init__(self, data_input, embd_sz = 30, num_epoch=1, batch_sz = 20):
		super(BigramInNN, self).__init__()
		self.num_epoch = num_epoch
		self.batch_sz = batch_sz
		self.data_input = data_input
		v_sz = data_input.get_vocab_size()

		# Parameters
		self.INPT = tf.placeholder(tf.int32, [None]) 
		self.OUT = tf.placeholder(tf.int32, [None])
		self.e = self.random_variable([v_sz, embd_sz])
		self.embd = tf.nn.embedding_lookup(self.e, self.INPT)

		self.w1 = self.weight_variable([embd_sz, 100])
		self.b1 = self.bias_variable([100])
		self.w2 = self.weight_variable([100, v_sz])
		self.b2 = self.bias_variable([v_sz])

		self.h1 = tf.nn.relu(tf.matmul(self.embd, self.w1) + self.b1)
		self.h2 = tf.matmul(self.h1, self.w2) + self.b2

		# Error and training step
		self.error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.h2, self.OUT))
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.error)

		# Evaluation steps
		self.correct_prediction = tf.equal(tf.argmax(self.h2,1), tf.cast(self.OUT, tf.int64))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		
		# Initialization
		init = tf.initialize_all_variables()
		self.sess = tf.Session()
		self.sess.run(init)


	def train(self):
		for time in xrange(self.num_epoch):
			count = 0
			for xs, ys in self.data_input.next_batch(self.batch_sz):
				self.sess.run(self.train_step, feed_dict = {self.INPT: xs, self.OUT: ys})
				error = self.sess.run(self.error, feed_dict = {self.INPT: xs, self.OUT: ys})
				training_accuracy = self.accuracy.eval(feed_dict = {self.INPT: xs, self.OUT: ys}, session = self.sess)
				print "step %d, training accuracy %g, perplexity %g"%(count, training_accuracy, math.exp(error))
				count += 1


	def test(self):
		total_perp = 0.0
		xs, ys = self.data_input.test_data()
		error = self.error.eval(feed_dict = {self.INPT: xs, self.OUT: ys}, session = self.sess)
		print "perplexity is %g"%(math.exp(error))


	def random_variable(self, shape):
		initial = tf.random_uniform(shape, -0.5, 0.5)
		return tf.Variable(initial)

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev = 0.1)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape = shape)
		return tf.Variable(initial)
		


if __name__ == '__main__':
	data_input = InputData('train.txt', 'test.txt')
	nn = BigramInNN(data_input)
	nn.train()
	nn.test()