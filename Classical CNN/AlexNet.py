"""
完整AlexNet.py网络结构，没有ILSVRC数据集处理方式
"""

#导入系统库及常用包
from datetime import datetime
import math
import time	
import tensorflow as tf
import numpy as np 
import cifar10, cifar10_input

batch_size = 128
max_step = 3000

#==============定义显示输出tensor的尺寸=================================
def print_activations(t):
	print(t.op.name, ' ', t.get_shape().as_list())



#=============数据集===================================================
#这里获取数据集，cifar10数据集由于图片太小，不改AlextNet网络参数没法使用
#正常ILSVRC图片数据集是224x224
# #使用cifar10类数据集,32x32分辨率图片,经过处理输出24x24大小的区块
# #data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'
# cifar10.maybe_download_and_extract()
# images_train, labels_train = cifar10_input.distorted_inputs(data_dir = data_dir, batch_size = batch_size)
# images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir, batch_size = batch_size)

# image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
# label_holder = tf.placeholder(tf.int32, [batch_size])
keep_prob = tf.placeholder(tf.float32)


#================AlexNet网络结构的Inference============================
parameters = []
#conv1
with tf.name_scope('conv1') as scope:
	kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype = tf.float32, stddev = 1e-1), name = 'weights')
	conv = tf.nn.conv2d(image_holder, kernel, [1, 4, 4, 1], padding = 'SAME')
	biases = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32), trainable = True, name = 'biases')
	bias = tf.nn.bias_add(conv, biases)
	conv1 = tf.nn.relu(bias, name = scope)
	print_activations(conv1)
	parameters += [kernel, biases]

#lrn1
lrn1 = tf.nn.lrn(conv1, 4, bias = 1.0, alpha = 0.001/9, beta = 0.75, name = 'lrn1')

#pool1
pool1 = tf.nn.max_pool(lrn1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool1')
print_activations(pool1)

#conv2
with tf.name_scope('conv2') as scope:
	kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype = tf.float32, stddev = 1e-1), name = 'weights')
	conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding = 'SAME')
	biases = tf.Variable(tf.constant(0.0, shape = [192], dtype = tf.float32), trainable = True, name = 'biases')
	bias = tf.nn.bias_add(conv, biases)
	conv2 = tf.nn.relu(bias, name = scope)
	parameters += [kernel, biases]
	print_activations(conv2)

#lrn2	
lrn2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha = 0.001/9, beta = 0.75, name = 'lrn2')

#pool2
pool2 = tf.nn.max_pool(lrn2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool2')
print_activations(pool2)

#conv3
with tf.name_scope('conv3') as scope:
	kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype = tf.float32, stddev = 1e-1), name = 'weights')
	conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding = 'SAME')
	biases = tf.Variable(tf.constant(0.0, shape = [384], dtype = tf.float32), trainable = True, name = 'biases')
	bias = tf.nn.bias_add(conv, biases)
	conv3 = tf.nn.relu(bias, name = scope)
	parameters += [kernel, biases]
	print_activations(conv3)

# conv4
with tf.name_scope('conv4') as scope:
	kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype = tf.float32, stddev = 1e-1), name = 'weights')
	conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding = 'SAME')
	biases = tf.Variable(tf.constant(0.0, shape = [256], dtype = tf.float32), trainable = True, name = 'biases')
	bias = tf.nn.bias_add(conv, biases)
	conv4 = tf.nn.relu(bias, name = scope)
	parameters += [kernel, biases]
	print_activations(conv4)

#conv5
with tf.name_scope('conv5') as scope:
	kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype = tf.float32, stddev = 1e-1), name = 'weights')
	conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding = 'SAME')
	biases = tf.Variable(tf.constant(0.0, shape = [256], dtype = tf.float32), trainable = True, name = 'biases')
	bias = tf.nn.bias_add(conv, biases)
	conv5 = tf.nn.relu(bias, name = scope)
	parameters += [kernel, biases]
	print_activations(conv5)

#pool5
pool5 = tf.nn.max_pool(conv5, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool5')
print_activations(pool5)

#fc6
with tf.name_scope('fc6') as scope:
	reshape = tf.reshape(pool5, [batch_size, -1])
	dim = reshape.get_shape()[1].value
	weight6 = tf.Variable(tf.truncated_normal([dim, 4096], dtype = tf.float32, stddev = 1e-1), name = 'weights')
	biases = tf.Variable(tf.zeros([4096]), name = 'biases')
	fc6 = tf.nn.relu(tf.matmul(reshape, weight6) + biases, name = scope)
	parameters += [weight6, biases]
	print_activations(fc6)

#dropout6
fc6__drop = tf.nn.dropout(fc6, keep_prob)

#fc7
with tf.name_scope('fc7') as scope:
	weight7 = tf.Variable(tf.truncated_normal([4096, 4096], dtype = tf.float32, stddev = 1e-1), name = 'weights')
	biases = tf.Variable(tf.zeros([4096]), name = 'biases')
	fc7 = tf.nn.relu(tf.matmul(fc6__drop, weight7) + biases, name = scope)
	parameters += [weight7, biases]
	print_activations(fc7)

#dropout7
fc7_drop = tf.nn.dropout(fc7, keep_prob)

#fc8
with tf.name_scope('fc8') as scope:
	weight8 = tf.Variable(tf.truncated_normal([4096, 1000], dtype = tf.float32, stddev = 1e-1), name = 'weights')
	biases = tf.Variable(tf.zeros([1000]), name = 'biases')
	fc8 = tf.matmul(fc7_drop, weight8) + biases
	parameters += [weight8, biases]
	print_activations(fc8)

def loss(logits, labels):
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels, name = 'cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
	return cross_entropy_mean



#================================训练==========================================================
#loss
loss = loss(fc8, label_holder)
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

#使用tf.nn.in_top_k函数求输出结果中top k的准确率，默认使用1
top_k_op = tf.nn.in_top_k(fc8, label_holder, 1)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

for step in range(max_step):
	start_time = time.time()
	#这里使用自定义方法获取训练集的image_batch和label_batch
	# image_batch, label_batch = sess.run([images_train, labels_train])
	_, loss_value = sess.run([train_step, loss], feed_dict = {image_holder: image_batch, label_holder: label_batch, keep_prob: 0.5})
	duration = time.time() - start_time
	if step % 10 == 0:
		examples_per_sec = batch_size / duration
		sec_per_batch = float(duration)
		format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
		print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))


#========================测试集测试================================================
num_examples = 10000
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
	#这里使用自定义方法获取测试集的image_batch和label_batch
	# image_batch, label_batch = sess.run([images_test, labels_test])
	predictions = sess.run([top_k_op], feed_dict = {image_holder: image_batch, label_holder: label_batch, keep_prob: 1.0})
	true_count += np.sum(predictions)
	step += 1

precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)