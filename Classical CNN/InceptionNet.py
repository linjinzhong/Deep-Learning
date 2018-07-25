"""Google Inception Net 首次出现在ILSVRC 2014的比赛中（和VGGNet同年），以较大优势取得第一：top-5错误率6.67%
通常被称为Inception V1， 22层深，比AlexNet的8层或者VGGNet的16层还要更深
参数量仅为500万，是AlexNet6000万的1/12
参数少效果好的原因除了用模型层数更深，表达呢能力更强外，还有两点：一是去除最后的全连接层，用全局平均池化层来取代它。
二是Inception V1中精心设计的Inception Module提高了参数的利用率。这两点借鉴了Network In Network论文
Google Inception Net是一个家族,V1(6.67%),V2(4.8%),V3(3.5%),V4(3.08%)
本节实现V3,top-5错误率3.5%
使用contrib.slim辅助设计这个网络，构建42层深的Inception V3
"""


import tensorflow as tf 
import time
import math
from datetime import datetime

#使用tf.contrib.slim辅助设计网络
slim = tf.contrib.slim

#定义产生截断的正态分布的简单函数
#lambda作为一个表达式，定义了一个匿名函数
#stddev为入口参数，:后面为函数体。在这里lambda简化了函数定义的书写形式
#使代码更为简洁，但是使用函数的定义方式更为直观，易理解
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)



#定义函数inception_v3_arg_scope,生成网络中经常用到的函数的默认参数
#epsilon:Epsilon是希腊语第五个字母艾普西隆的小写，写作ϵ或ε,数学参数，常用来表示小额度的，少量的
def inception_v3_arg_scope(weight_decay = 0.00004, stddev = 0.1, batch_norm_var_collection = 'moving_vars'):
	#定义batch normalization 的参数字典
	batch_norm_params = {
		'decay': 0.9997,
		'epsilon': 0.001,   
		'updates_collections': tf.GraphKeys.UPDATE_OPS,
		'variables_collections': {
			'beta': None,
			'gamma': None,
			'moving_mean': [batch_norm_var_collection],
			'moving_variance': [batch_norm_var_collection],
		}
	}

	#使用slim.arg_scope，可以给函数的参数自动赋予某些默认值
	with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer = slim.l2_regularizer(weight_decay)):
		with slim.arg_scope([slim.conv2d], weights_initializer = tf.truncated_normal_initializer(stddev = stddev), activation_fn = tf.nn.relu, normalizer_fn = slim.batch_norm, normalizer_params = batch_norm_params) as sc:
			return sc




#定义函数inception_v3_base,生成Inception V3网络的卷积部分
#inputs:输入图片数据的tensor
#scope:包含函数默认参数的环境
def inception_v3_base(inputs, scope = None):
	#定义字典表，用来保存某些关键节点供之后使用
	end_points = {}

	with tf.variable_scope(scope, 'InceptionV3', [inputs]):

		#5个卷积层，2个池化层，实现对输入图片数据的尺寸压缩，并对图片特征进行抽象
			#输入数据尺寸为299x299x3,输出35x35x192
		with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding = 'VALID'):
			net = slim.conv2d(inputs, 32, [3, 3], stride = 2, scope = 'Conv2d_1a_3x3')
			net = slim.conv2d(net, 32, [3, 3], scope = 'Conv2d_2a_3x3')
			net = slim.conv2d(net, 64, [3, 3], padding = 'SAME', scope = 'Conv2d_2b_3x3')
			net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'MaxPool_3a_3x3')
			net = slim.conv2d(net, 80, [1, 1], scope = 'Conv2d_3b_1x1')
			net = slim.conv2d(net, 192, [3, 3], scope = 'Conv2d_4a_3x3')
			net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'MaxPool_5a_3x3')


		#=============连续的三个Inception模块组=================
		with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding = 'SAME'):

		#第1个Inception模块组包含3个结构相似的Inception Module
			
			#第一个Inception Module——Mixed_5b
			#输出为35x35x(64+64+96+32)=35x35x256
			with tf.variable_scope('Mixed_5b'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 48, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope = 'Conv2d_0b_5x5')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0b_3x3')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0c_3x3')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope = 'Conv2d_0b_1x1')
				#在第三维（即通道数）上合并
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

			#第2个Inception Module——Mixed_5c
			#和第一个除了最后第4个分支的32改为64外都相同
			##输出为35x35x(64+64+96+64)=35x35x288
			with tf.variable_scope('Mixed_5c'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 48, [1, 1], scope = 'Conv2d_0b_1x1')
					branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope = 'Conv_1_0c_5x5')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0b_3x3')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0c_3x3')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope = 'Conv2d_0b_1x1')
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

			#第3个Inception Module———Mixed_5d
			#和第2个完全相同
			##输出为35x35x(64+64+96+32)=35x35x288
			with tf.variable_scope('Mixed_5d'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 48, [1, 1], scope = 'Conv2d_0b_1x1')
					branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope = 'Conv_1_0c_5x5')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0b_3x3')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0c_3x3')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope = 'Conv2d_0b_1x1')
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)


		#第2个Inception模块组包含5个结构相似的Inception Module,输出尺寸全部定格为17x17x768
			#第1个Inception Module——Mixed_6a
			#输出为17x17x(384+96+288)=17x17x768
			with tf.variable_scope('Mixed_6a'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 384, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope = 'Conv2d_0b_3x3')
					branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1a_1x1')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.max_pool2d(net, [3, 3], stride = 2, padding = 'VALID', scope = 'MaxPool_1a_3x3')
				net = tf.concat([branch_0, branch_1, branch_2], 3)

			#第2个Inception Module——Mixed_6b
			#输出为17x17x(192+192+192+192)=17x17x768
			with tf.variable_scope('Mixed_6b'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 128, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope = 'Conv2d_0b_1x7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 128, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope = 'Conv2d_0b_7x1')
					branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope = 'Conv2d_0c_1x7')
					branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope = 'Conv2d_0d_7x1')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope = 'Conv2d_0e_1x7')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

			#第3个Inception Module——Mixed_6c
			#和前面第2个非常相似，只有的第2个分支和第3个分支中前几个卷积层的输出通道不同，从128变为了160
			#输出为17x17x(192+192+192+192)=17x17x768
			with tf.variable_scope('Mixed_6c'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope = 'Conv2d_0b_1x7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 128, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0b_7x1')
					branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope = 'Conv2d_0c_1x7')
					branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0d_7x1')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope = 'Conv2d_0e_1x7')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

			#第4个Inception Module——Mixed_6d
			#和前面第3个Mixed_6c完全相同
			#输出为17x17x(192+192+192+192)=17x17x768
			with tf.variable_scope('Mixed_6d'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope = 'Conv2d_0b_1x7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 128, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0b_7x1')
					branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope = 'Conv2d_0c_1x7')
					branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0d_7x1')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope = 'Conv2d_0e_1x7')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

			#第5个Inception Module——Mixed_6e
			#和前面第3/4个Mixed_6c,Mixed_6d完全相同
			#输出为17x17x(192+192+192+192)=17x17x768
			with tf.variable_scope('Mixed_6e'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope = 'Conv2d_0b_1x7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 128, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0b_7x1')
					branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope = 'Conv2d_0c_1x7')
					branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0d_7x1')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope = 'Conv2d_0e_1x7')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
			#将Mixe_6e存储在end_points中，作为Auxiliary classifier辅助模型的分类
			end_points['Mixed_6e'] = net


		#第3个Inception模块组包含3个结构相似的Inception Module
			#第1个Inception Module——Mixed_7a
			#输出为8x8x(320+192+768)=8x8x1280
			with tf.variable_scope('Mixed_7a'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1a_3x3')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope = 'Conv2d_0b_1x7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
					branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1a_3x3')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.max_pool2d(net, [3, 3], stride = 2, padding = 'VALID', scope = 'MaxPool_1a_3x3')
				net = tf.concat([branch_0, branch_1, branch_2], 3)

			#第2个Inception Module——Mixed_7b
			#输出为8x8x(320+768+768+192)=8x8x2048
			with tf.variable_scope('Mixed_7b'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 320, [1, 1], scope= 'Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 384, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope = 'Conv2d_0b_1x3'), slim.conv2d(branch_1, 384, [3, 1], scope = 'Conv2d_0b_3x1')], 3)
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 448, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope = 'Conv2d_0b_3x3')
					branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope = 'Conv2d_0c_1x3'), slim.conv2d(branch_2, 384, [3, 1], scope = 'Conv2d_0d_3x1')], 3)
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

			#第3个Inception Module——Mixed_7c
			#和前面的Mixed_7b完全一致
			#输出为8x8x(320+768+768+192)=8x8x2048
			with tf.variable_scope('Mixed_7c'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 320, [1, 1], scope= 'Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 384, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope = 'Conv2d_0b_1x3'), slim.conv2d(branch_1, 384, [3, 1], scope = 'Conv2d_0b_3x1')], 3)
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 448, [1, 1], scope = 'Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope = 'Conv2d_0b_3x3')
					branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope = 'Conv2d_0c_1x3'), slim.conv2d(branch_2, 384, [3, 1], scope = 'Conv2d_0d_3x1')], 3)
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
			return net, end_points


#========最后一部分：全局平均池化、Softmax和Auxiliary Logits==========================
def inception_v3(inputs, num_classes = 1000, is_training = True, dropout_keep_prob = 0.8, prediction_fn = slim.softmax,	spatial_squeeze = True, reuse = None, scope = 'InceptionV3'):
	with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse = reuse) as scope:
		with slim.arg_scope([slim.batch_norm, slim.dropout], is_training = is_training):
			net, end_points = inception_v3_base(inputs, scope = scope)

			#处理Auxiliary Logits这部分的逻辑
			with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding = 'SAME'):
				aux_logits = end_points['Mixed_6e']
				with tf.variable_scope('AuxLogits'):
					aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride = 3, padding = 'VALID', scope = 'AvgPool_1a_5x5')
					aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope = 'Conv2d_1b_1x1')
					aux_logits = slim.conv2d(aux_logits, 768, [5, 5], weights_initializer = trunc_normal(0.01), padding = 'VALID', scope = 'Conv2d_2a_5x5')
					aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn = None, normalizer_fn = None, weights_initializer = trunc_normal(0.001), scope = 'Conv2d_2b_1x1')
					if spatial_squeeze:
						aux_logits = tf.squeeze(aux_logits, [1, 2], name = 'SpatialSqueeze')
					end_points['AuxLogits'] = aux_logits

			#处理正常的分类逻辑
			with tf.variable_scope('Logits'):
				net = slim.avg_pool2d(net, [8, 8], padding = 'VALID', scope = 'AvgPool_1a_8x8')
				net = slim.dropout(net, keep_prob = dropout_keep_prob, scope = 'Dropout_1b')
				end_points['PreLogits'] = net
				logits = slim.conv2d(net, num_classes, [1, 1], activation_fn = None, normalizer_fn = None, scope= 'Conv2d_1c_1x1')
				if spatial_squeeze:
					logits = tf.squeeze(logits, [1, 2], name = 'SpatialSqueeze')
			end_points['Logits'] = logits
			end_points['Predictions'] = prediction_fn(logits, scope = 'Predictions')
	return logits, end_points



#=================性能测试=======================
#评测函数
def time_tensorflow_run(session, target, info_string):
	num_steps_burn_in = 10
	total_duration = 0.0
	total_duration_squared = 0.0

	for i in range(num_batches + num_steps_burn_in):
		start_time = time.time()
		_ = session.run(target)
		duration = time.time() - start_time
		if i >= num_steps_burn_in:
			if not i % 10:
				print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
			total_duration +=duration
			total_duration_squared += duration * duration

	mn = total_duration / num_batches
	vr = total_duration_squared / num_batches - mn * mn
	sd = math.sqrt(vr)
	print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd))


batch_size = 32
height, width = 299, 299
inputs = tf.random_uniform((batch_size, height, width, 3))
with slim.arg_scope(inception_v3_arg_scope()):
	logits, end_points = inception_v3(inputs, is_training = False)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches = 100
time_tensorflow_run(sess, logits, 'Forward')
