"""牛津大学计算机视觉组和Google DeepMind公司研究员一起研发
通过反复堆叠3x3的小型卷积核和2x2的最大池化层，构筑16~19层深的卷积神经网络
以下是VGGNet-16代码
"""
from datetime import datetime
import math
import time
import tensorflow as tf 


#定义卷积层
#input_op:输入的tensor
#name:这层的名称
#kh,kw:卷积核的高和宽
#n_out:卷积核数量即输出通道数
#dh,dw:步长的高和宽
#p:参数列表
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
	#获取输入input_op的通道数
	n_in = input_op.get_shape()[-1].value

	with tf.name_scope(name) as scope:
		#Xaiver初始化器会根据某一层网络的输入、输出结点数量自动调整最合适的分布
		kernel = tf.get_variable(scope + "w", shape = [kh, kw, n_in, n_out], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer_conv2d())
		conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding = 'SAME')
		bias_init_val = tf.constant(0.0, shape = [n_out], dtype = tf.float32)
		biases = tf.Variable(bias_init_val, trainable = True, name = 'b')
		z = tf.nn.bias_add(conv, biases)
		activation = tf.nn.relu(z, name = scope)
		p += [kernel, biases]
		return activation



#定义全连接层的创建函数fc_op
#input_op:输入
#name:该层名称
#n_out:输出通道数
#p:参数列表
def fc_op(input_op, name, n_out, p):
	#获得输入input_op的通道数
	n_in = input_op.get_shape()[-1].value

	with tf.name_scope(name) as scope:
		kernel = tf.get_variable(scope + "w", shape = [n_in, n_out], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
		biases = tf.Variable(tf.constant(0.1, shape = [n_out], dtype = tf.float32), name = 'b')
		activation = tf.nn.relu_layer(input_op, kernel, biases, name = scope)
		p += [kernel, biases]
		return activation



#定义最大池化层的创建函数mpool_op
#input_op:输入
#name:该层名称
#kh,kw:池化尺寸
#dh,dw:步长
def mpool_op(input_op, name, kh, kw, dh, dw):
	return tf.nn.max_pool(input_op, ksize = [1, kh, kw, 1], strides = [1, dh, dw, 1], padding = 'SAME', name = name)
 

#======创建VGGNet-16的网络结构======================
#VGGNet-16主要分为6个部分，前5段为卷积网络，最后一段是全连接网络

#定义创建VGGNe-16网络结构的函数inference_op
def inference_op(input_op, keep_prob):
	#初始化参数列表
	p = []

	#第1段，2个卷积层一个池化层
	#input_op:224x224x3
	#pool1:112x112x64
	conv1_1 = conv_op(input_op, name = "conv1_1", kh = 3, kw = 3, n_out = 64, dh = 1, dw = 1, p = p)
	conv1_2 = conv_op(conv1_1, name = "conv1_2", kh = 3, kw = 3, n_out = 64, dh = 1, dw = 1, p = p)
	pool1 = mpool_op(conv1_2, name = "pool1", kh = 2, kw = 2, dh = 2, dw = 2)

	#第2段，2个卷积层一个池化层
	#pool2:56x56x128
	conv2_1 = conv_op(pool1, name = "conv2_1", kh = 3, kw = 3, n_out = 128, dh = 1, dw = 1, p = p)	
	conv2_2 = conv_op(conv2_1, name = "conv2_2", kh = 3, kw = 3, n_out = 128, dh = 1, dw = 1, p = p)	
	pool2 = mpool_op(conv2_2, name = "pool2", kh = 2, kw = 2, dh = 2, dw = 2)

	#第3段，3个卷积层一个池化层
	#pool3:28x28x256
	conv3_1 = conv_op(pool2, name = "conv3_1", kh = 3, kw = 3, n_out = 256, dh = 1, dw = 1, p = p)	
	conv3_2 = conv_op(conv3_1, name = "conv3_2", kh = 3, kw = 3, n_out = 256, dh = 1, dw = 1, p = p)	
	conv3_3 = conv_op(conv3_2, name = "conv3_3", kh = 3, kw = 3, n_out = 256, dh = 1, dw = 1, p = p)	
	pool3 = mpool_op(conv3_3, name = "pool3", kh = 2, kw = 2, dh = 2, dw = 2)

	#第4段，3个卷积层一个池化层
	#pool4:14x14x512
	conv4_1 = conv_op(pool3, name = "conv4_1", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)	
	conv4_2 = conv_op(conv4_1, name = "conv4_2", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)	
	conv4_3 = conv_op(conv4_2, name = "conv4_2", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)	
	pool4 = mpool_op(conv4_3, name = "pool4", kh = 2, kw = 2, dh = 2, dw = 2)

	#第5段，3个卷积层一个池化层
	#pool5:7x7x512
	conv5_1 = conv_op(pool4, name = "conv5_1", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)	
	conv5_2 = conv_op(conv5_1, name = "conv5_2", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)	
	conv5_3 = conv_op(conv5_2, name = "conv5_3", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)	
	pool5 = mpool_op(conv5_3, name = "pool5", kh = 2, kw = 2, dh = 2, dw = 2)

	#将第5段输出进行扁平化，形成7x7x512=25088的一维向量
	shp = pool5.get_shape()
	flattened_shap = shp[1].value * shp[2].value * shp[3].value
	resh1 = tf.reshape(pool5, [-1, flattened_shap], name = "resh1")

	#然后连接一个隐含节点数为4096的全连接层，激活函数为ReLU,然后连接一个Dropout层，训练时保留率为0.5，预测时为1.0
	fc6 = fc_op(resh1, name = "fc6", n_out = 4096, p = p)
	fc6_drop = tf.nn.dropout(fc6, keep_prob, name = "fc6_drop")

	#和上面一样的全连接层
	fc7 = fc_op(fc6_drop, name = "fc7", n_out = 4096, p = p)
	fc7_drop = tf.nn.dropout(fc7, keep_prob, name = "fc7_drop")

	#最后连接一个1000个输出结点的全连接层，并使用softmax
	fc8 = fc_op(fc7_drop, name = "fc8", n_out = 1000, p = p)
	softmax = tf.nn.softmax(fc8)
	predictions = tf.argmax(softmax, 1)
	return predictions, softmax, fc8, p


#评测函数
def time_tensorflow_run(session, target, feed, info_string):
	num_step_burn_in = 10
	total_duration = 0.0
	total_duration_squared = 0.0

	for i in range(num_batches + num_step_burn_in):
		start_time = time.time()
		_ = session.run(target, feed_dict = feed)
		duration = time.time() - start_time
		if i >= num_step_burn_in:
			if not i % 10:
				print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_step_burn_in, duration))
			total_duration +=duration
			total_duration_squared += duration * duration

	mn = total_duration / num_batches
	vr = total_duration_squared / num_batches - mn * mn
	sd = math.sqrt(vr)
	print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd))

	
#评测函数的主函数
def run_benchmark():
	with tf.Graph().as_default():
		image_size = 224
		images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype = tf.float32, stddev = 1e-1))

		keep_prob = tf.placeholder(tf.float32)
		predictions, softmax, fc8, p = inference_op(images, keep_prob)

		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)

		time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, "Forward")
		objective = tf.nn.l2_loss(fc8)
		grad = tf.gradients(objective, p)
		time_tensorflow_run(sess, grad, {keep_prob: 0.5}, "Forward-backward")


#主函数
batch_size = 32
num_batches = 1000
run_benchmark()