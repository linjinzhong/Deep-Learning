"""
使用LSTM实现一个语言模型
"""

# #首先下载PTB数据集并解压
# #该数据集已经做了预处理，包含1万个不同的单词，有句尾的标记，同时将罕见的词汇统一处理为特殊字符
# wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
# tar xvf simple-examples.tgz

# #下载TensorFlow Models库，进入目录models/tutorials/rnn/ptb
# #然后载入常用的库和TensorFlow Models中的PTB reader，借助它读取数据内容
# #读取数据内容比较繁琐，主要是将单词转为唯一的数字编码，一遍神经网络处理
# git clone https://github.com/tensorflow/models.git
# cd models/tutorials/rnn/ptb

import time
import numpy as np 
import tensorflow as tf 
import reader


#定义语言模型处理输入数据的class, PTBInput
class PTBInput(object):
	def __init__(self, config, data, name = None):
		self.batch_size = batch_size = config.batch_size
		#LSTM展开步数
		self.num_steps = num_steps = config.num_steps
		#计算每个epoch的size，即每个epoch内需要多少轮训练的迭代
		self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
		#获取特征数据input_data以及label数据targets
		self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name = name)



#定义语言模型的class, PTBModel
class PTBModel(object):
	def __init__(self, is_training, config, input_):
		self._input = input_
		batch_size = input_.batch_size
		num_steps = input_.num_steps
		#hidden_size是LSTM的节点数
		size = config.hidden_size
		#vocab_size是词汇表的大小
		vocab_size = config.vocab_size

		#使用tf.contrib.rnn.BasicLSTMCell设置我们默认的LSTM单元
		def lstm_cell():
			return tf.contrib.rnn.BasicLSTMCell(size, forget_bias = 0.0, state_is_tuple = True)

		attn_cell = lstm_cell
		if is_training and config.keep_prob < 1:
			def attn_cell():
				return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob = config.keep_prob)
		#RNN堆叠函数将前面构造的lstm_cell多层堆叠得到cell
		cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple = True)
		#设置LSTM单元的初始化状态为0
		self._initial_state = cell.zero_state(batch_size, tf.float32)

		#创建网络的词嵌入部分
		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [vocab_size, size], dtype = tf.float32)
			inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

		if is_training and config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs, config.keep_prob)

		#定义输出outputs
		outputs = []
		state = self._initial_state
		with tf.variable_scope("RNN"):
			for time_step in range(num_steps):
				if time_step > 0: 
					tf.get_variable_scope().reuse_variables()
				#inputs[:, time_step, :]代表所欲样本的第time_step个单词，第3个维度是单词的向量表达的维度
				(cell_output, state) = cell(inputs[:, time_step, :], state)
				outputs.append(cell_output)

		#将output内容用tf.concat串接到一起
		output = tf.reshape(tf.concat(outputs, 1), [-1, size])
		softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype = tf.float32)
		softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype = tf.float32)
		logits = tf.matmul(output, softmax_w) + softmax_b
		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(input_.targets, [-1])], [tf.ones([batch_size * num_steps], dtype = tf.float32)])
		self._cost = cost = tf.reduce_sum(loss) / batch_size
		self._final_state = state

		#不是训练状态则直接返回
		if not is_training:
			return

		#定义学习速率的变量_lr
		self._lr = tf.Variable(0.0, trainable = False)
		#获取全部可训练的参数
		tvars = tf.trainable_variables()
		#针对前面的cost计算tvars的梯度, 设置梯度的最大范数
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step = tf.contrib.framework.get_or_create_global_step())

		#设置控制学习速率的placeholder
		self._new_lr = tf.placeholder(tf.float32, shape = [], name = "new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)

	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict = {self._new_lr: lr_value})



#定义PTBModel classd的一些preoperty
#python中@property装饰器可以将返回变量设为只读，防止修改变量引发的问题
@property
def input(self):
	return self._input

@property
def initial_state(self):
	return self._initial_state

@property
def cost(self):
	return self._cost

@property
def final_state(self):
	return self._final_state

@property
def lr(self):
	return self._lr

@property
def train_op (self):
	return self._train_op 


#定义几种大小不同的模型参数
class SmallConfig(object):
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm =  5
	num_layers = 2
	num_steps = 20
	hidden_size = 200
	max_epoch = 4
	max_max_epoch = 13
	keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 10000

class MediumConfig(object):
	init_scale = 0.05 
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2 
	num_steps = 35
	hidden_size = 650
	max_epoch = 6 
	max_max_epoch = 39
	keep_prob = 0.5
	lr_decay = 0.8
	batch_size = 20 
	vocab_size = 10000

class LargeConfig(object):
	init_scale = 0.04
	learning_rate = 1.0
	max_grad_norm = 10 
	num_layers = 2 
	num_steps = 35 
	hidden_size = 1500 
	max_epoch = 14
	max_max_epoch = 55
	keep_prob = 0.35 
	Lr_decay = 1 / 1.15
	batch_size = 20 
	vocab_size = 10000

#测试
class TestConfig(object):
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 1 
	num_layers = 1 
	num_steps = 2 
	hidden_size = 2 
	max_epoch = 1 
	max_max_epoch = 1 
	keep_prob = 1.0 
	lr_decay = 0.5 
	batch_size = 20
	vocab_size = 10000


#定义训练一个epoch数据的函数run_epoch
def run_epoch(session, model, eval_op = None, verbose = False):
	start_time = time.time()
	costs = 0.0 
	iters = 0 
	state = session.run(model.initial_state)

	fetches = {
	"cost": model.cost, "final_state": model.final_state,
	}

	if eval_op is not None:
		fetches["eval_op"] = eval_op

	for step in range(model.input.epoch_size):
		feed_dict = {}
		for i, (c, h) in enumerate(model.initial_state):
			feed_dict[c] = state[i].c 
			feed_dict[h] = state[i].h 

		vals = session.run(fetches, feed_dict)
		cost = vals["cost"]
		state = vals["final_state"]

		costs += cost
		iters += model.input.num_steps

		if verbose and step % (model.input.epoch_size // 10) == 10:
			print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / model.input.epoch_size, np.exp(costs / iters), iters * model.input.batch_size / (time.time() - start_time)))

	return np.exp(costs / iters)

#读取数据
raw_data = reader.ptb_raw_data('simple-examples/data/')
train_data, valid_data, test_data, _ = raw_data
config = SmallConfig()
eval_config = SmallConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1 


#创建默认的Graph
with tf.Graph().as_default():
	initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

	with tf.name_scope("Train"):
		train_input = PTBInput(config = config, data = train_data, name = "TrainInput")
		with tf.variable_scope("Model", reuse = None, initializer = initializer):
			m = PTBModel(is_training = True, config = config, input_ = train_input)

	with tf.name_scope("Valid"):
		valid_input = PTBInput(config = config, data = valid_data, name = "ValidInput")
		with tf.variable_scope("Model", reuse = True, initializer = initializer):
			mvalid = PTBModel(is_training = False, config = config, input_ = valid_input)

	with tf.name_scope("Test"):
		test_input = PTBInput(config = config, data = test_data, name = "TestInput")
		with tf.variable_scope("Model", reuse = True, initializer = initializer):
			mtest = PTBModel(is_training = False, config = eval_config, input = test_input)


#使用tf.train.Supervisor()创建训练的管理器sv
sv = tf.train.Supervisor()
with sv.managed_session() as session:
	for i in range(config.max_max_epoch):
		lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
		m.assign_lr(session, config.learning_rate * lr_decay)

		print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
		train_perplexity = run_epoch(session, m, eval_op = m.train_op, verbose = True)
		print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
		valid_perplexity = run_epoch(session, mvalid)
		print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

	test_perplexity = run_epoch(session, mtest)
	pritn("Test Perplexity: %.3f" % test_perplexity)


