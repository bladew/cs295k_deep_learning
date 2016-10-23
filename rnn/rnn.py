import re
import tensorflow as tf
import math


class DataProcessor(object):
	def __init__(self, book_path):
		super(DataProcessor, self).__init__()
		end_point = set(['.', '!', '?'])
		tokens = {'eos':0}
		source = []
		with open(book_path, 'r') as book:
			for line in book:
				for token in self.basic_tokenizer(line):
					source.append(token)
					if token not in tokens:
						tokens[token] = 0
					tokens[token] += 1
					if token in end_point:
						tokens['eos'] += 1

		# Cut down vocabulary size.
		vocab = sorted(tokens.items(), key = lambda x:-x[1])
		size = 8000
		self.tokens_id = {"UNK":0}
		for i, item in enumerate(vocab):
			if i <= size:
				self.tokens_id[item[0]] = i + 1
			else:
				break

		# Divide into training and testing data.
		self.train_data = [self.tokens_id[e] if e in self.tokens_id else self.tokens_id["UNK"] for e in source[:int(0.9 * len(source))]]
		self.test_data = [self.tokens_id[e] if e in self.tokens_id else self.tokens_id["UNK"] for e in source[int(0.9 * len(source)):]]


	def next_input(self, num_step, batch_sz, test = False):
		if test:
			data = self.test_data
		else:
			data = self.train_data

		xs, ys = [], []
		for x, y in self._next_batch(data, num_step):
			if len(x) == num_step:
				xs.append(x)
				ys.append(y)
			if len(xs) == batch_sz:
				res_x, res_y = xs, ys
				xs, ys = [], []
				yield res_x, res_y


	def _next_batch(self, data, size):
		i = 0
		while i + 1 < len(data):
			end = min(len(data), i + size)
			Xs, Ys = [], [] 
			while i < end and i + 1 < len(data):
				Xs.append(data[i])
				Ys.append(data[i + 1])
				i += 1
			yield Xs, Ys


	def get_vocab_size(self):
		return len(self.tokens_id)


	def basic_tokenizer(self, sentence, word_split=re.compile(b"([.,!?\"':;)(])")):
	    words = []
	    for space_separated_fragment in sentence.strip().split():
	    	words.extend(re.split(word_split, space_separated_fragment))
	    return [w.lower() for w in words if w]


class Rnn(object):
	def __init__(self, data_input, embd_sz = 50, num_epoch=5, state_sz = 256, num_step = 20, batch_sz = 20):
		super(Rnn, self).__init__()
		vocab_sz = data_input.get_vocab_size()
		self.batch_sz = batch_sz
		self.num_step = num_step
		self.num_epoch = num_epoch
		self.data_input = data_input

		# Placeholders.
		self.input = tf.placeholder(tf.int32, [batch_sz, num_step])
		self.output = tf.placeholder(tf.int32, [batch_sz, num_step])
		self.keep = tf.placeholder(tf.float32)

		# Parameters.
		self.e = self.random_variable([vocab_sz, embd_sz])
		self.lstm = tf.nn.rnn_cell.BasicLSTMCell(state_sz)
		self.init_s = self.lstm.zero_state(batch_sz, tf.float32)

		self.w1 = self.weight_variable([state_sz, vocab_sz])
		self.b1 = self.bias_variable([vocab_sz])

		# Feed forward pass.
		self.embd = tf.nn.embedding_lookup(self.e, self.input)
		self.embd_dropout = tf.nn.dropout(self.embd, self.keep) # Training only
		rnn_out, self.new_s = tf.nn.dynamic_rnn(self.lstm, self.embd_dropout, initial_state = self.init_s)
		self.rnn_out = tf.reshape(rnn_out, [-1, state_sz])
		self.logits = tf.matmul(self.rnn_out, self.w1) + self.b1

		# Error and training step
		self.loss = tf.nn.seq2seq.sequence_loss_by_example([self.logits], [tf.reshape(self.output, [-1])], [tf.ones([batch_sz * num_step], dtype=tf.float32)])
		self.average_loss = tf.reduce_sum(self.loss) / batch_sz
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.average_loss)

		# Evaluation steps
		self.correct_prediction = tf.equal(tf.argmax(self.logits,1), tf.cast(tf.reshape(self.output, [-1]), tf.int64))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		
		# Initialization
		init = tf.initialize_all_variables()
		self.sess = tf.Session()
		self.sess.run(init)


	def train(self):
		for time in xrange(self.num_epoch):
			iters = 0
			loss = 0.0
			init_s = self.sess.run(self.init_s)
			for xs, ys in self.data_input.next_input(self.num_step, self.batch_sz):
				loss_cur, init_s, accuracy, _ = self.sess.run([self.average_loss, self.new_s, self.accuracy, self.train_step], feed_dict = {self.input: xs, self.output: ys, self.keep: 0.5, self.init_s: init_s})
				loss += loss_cur
				iters += self.num_step
				print "step %d, accuracy %g, perplexity %g"%(iters/self.num_step, accuracy, math.exp(loss / iters))


	def test(self):
		iters = 0
		loss = 0.0
		init_s = self.sess.run(self.init_s)
		for xs, ys in self.data_input.next_input(self.num_step, self.batch_sz, test = True):
			loss_cur, init_s = self.sess.run([self.average_loss, self.new_s], feed_dict = {self.input: xs, self.output: ys, self.keep: 1, self.init_s: init_s})
			loss += loss_cur
			iters += self.num_step
		print "perplexity %g"%(math.exp(loss / iters))


	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev = 0.1)
		return tf.Variable(initial)


	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape = shape)
		return tf.Variable(initial)


	def random_variable(self, shape):
		initial = tf.random_uniform(shape, -1, 1)
		return tf.Variable(initial)


if __name__ == '__main__':
	data_input = DataProcessor('warandpeace.txt')
	nn = Rnn(data_input)
	nn.train()
	nn.test()
