"""ResNet(Residual Neural Network)由微软研究院的Kaiming He等4名华人提出，通过使用Residual Unit成功训练152层深的神经网络，
在ILSVRC 2015比赛中获得了冠军，取得了3.57%的top-5错误率，同时参数量比VGGNet低，效果非常突出。
使用TensorFlow实现一个ResNet V2网络
"""
import time
import math
import collections
import tensorflow as tf 
from datetime import datetime
slim = tf.contrib.slim


#example:Block('block1', bottleneck, [(256, 64, 1)] x 2 + [(256, 64 2)])
##block1是这个Block的name(或scope)，bottleneck是ResNet V2中的残差学习单元，最后args是参数列表
#args参数列表中每个元素对应一个bottleneck残差学习单元,每个元素都是一个三元tuple(depth,depth_bottleck, stride)
#比如(256, 64, 3),代表构建的bottleneck残差学习单元(每个残差学习单元包含三个卷积层)中，
#第三层输出通道数depth为256，前两层输出通道数depth_bottleneck为64，且中间那层步长为3
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
	'A named tuple describing a ResNet block.'


#定义降采样subsample的方法，参数包括inputs(输入)，factor(采样因子)和scope
#如果factor为1，直接返回inputs，如果不为1，使用slim.max_pool2d最大池化来实现
#通过1x1的池化尺寸，stride作步长，即可实现降采样
def subsample(inputs, factor, scope = None):
	if factor == 1:
		return inputs
	else:
		return slim.max_pool2d(inputs, [1, 1], stride = factor, scope = scope)



#定义一个conv2d_same函数创建卷积层
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope = None):
	if stride == 1:
		return slim.conv2d(inputs, num_outputs, kernel_size, stride = 1, padding = 'SAME', scope = scope)
	else:
		pad_total = kernel_size - 1
		pad_beg = pad_total // 2
		pad_end = pad_total - pad_beg
		inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
		return slim.conv2d(inputs, num_outputs, kernel_size, stride = stride, padding = 'VALID', scope = scope)



#定义堆叠Blocks的函数
@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections = None):
	for block in blocks:
		with tf.variable_scope(block.scope, 'block', [net]) as sc:
			for i, unit in enumerate(block.args):
				with tf.variable_scope('unit_%d' % (i + 1), values = [net]):
					unit_depth, unit_depth_bottleneck, unit_stride = unit
					net = block.unit_fn(net, depth = unit_depth, depth_bottleneck = unit_depth_bottleneck, stride = unit_stride)
				net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

	return net



#创建ResNet通用的arg_scope
def resnet_arg_scope(is_training = True, weight_decay = 0.0001, batch_norm_decay = 0.997, batch_norm_epsilon = 1e-5, batch_norm_scale = True):
	batch_norm_params = {'is_training': is_training, 'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale, 'updates_collections': tf.GraphKeys.UPDATE_OPS,}

	with slim.arg_scope([slim.conv2d], weights_regularizer = slim.l2_regularizer(weight_decay),	weights_initializer = slim.variance_scaling_initializer(), activation_fn = tf.nn.relu, normalizer_fn = slim.batch_norm, normalizer_params = batch_norm_params):
		with slim.arg_scope([slim.batch_norm], **batch_norm_params):
			with slim.arg_scope([slim.max_pool2d], padding = 'SAME') as arg_sc:
				return arg_sc



#定义核心的bottleneck残差学习单元
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections = None, scope = None):
	with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
		depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank = 4)
		preact = slim.batch_norm(inputs, activation_fn = tf.nn.relu, scope = 'preact')
		if depth == depth_in:
			shortcut = subsample(inputs, stride, 'shortcut')
		else:
			shortcut = slim.conv2d(preact, depth, [1, 1], stride = stride, normalizer_fn = None, activation_fn = None, scope = 'shortcut')
		residul = slim.conv2d(preact, depth_bottleneck, [1, 1], stride = 1, scope = 'conv1')			
		residul = conv2d_same(residul, depth_bottleneck, 3, stride = stride, scope = 'conv2')
		residul = slim.conv2d(residul, depth, [1, 1], stride = 1, normalizer_fn = None, activation_fn = None, scope = 'conv3')
		output = shortcut + residul
		return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)



#定义生成ResNet V2的主函数
def resnet_v2(inputs, blocks, num_classes = None, global_pool = True, include_root_block = True, reuse = None, scope = None):
	with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse = reuse) as sc:
		end_points_collection = sc.original_name_scope + '_end_points'
		with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense], outputs_collections = end_points_collection):
			net = inputs
			if include_root_block:
				with slim.arg_scope([slim.conv2d], activation_fn = None, normalizer_fn = None):
					net = conv2d_same(net, 64, 7, stride = 2, scope = 'conv1')
				net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'pool1')
			net = stack_blocks_dense(net, blocks)
			net = slim.batch_norm(net, activation_fn = tf.nn.relu, scope = 'postnorm')
			if global_pool:
				net = tf.reduce_mean(net, [1, 2], name = 'pool5', keep_dims = True)
			if num_classes is not None:
				net = slim.conv2d(net, num_classes, [1, 1], activation_fn = None, normalizer_fn = None, scope = 'logits')
			end_points = slim.utils.convert_collection_to_dict(end_points_collection)
			if num_classes is not None:
				end_points['predictions'] = slim.softmax(net, scope = 'predictions')
			return net, end_points





#================根据不同深度的ResNet网络配置，设计层数分别为50,101,152,200的ResNet================
#首先是50层的ResNet，4个残差学习的Blocks的units数量分别为3,4,6,3，总层数即为(3+4+6+3)x3+2=50
#残差学习模块之前的卷积和池化已经将尺寸缩小了4倍，前3个Blocks又都包含步长为2的层，因此总尺寸缩小了4x2^3=32倍， 输入图片尺寸最后变为224/32=7.
#和Inception V3很像，ResNet不断使用步长为2的层来缩减尺寸，但同时输出通道数也在持续增加，最后达到了2048
def resnet_v2_50(inputs, num_classes = None, global_pool = True, reuse = None, scope = 'resnet_v2_50'):
	blocks = [Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]), 
			  Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
			  Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
			  Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
	return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block = True, reuse = reuse, scope = scope)

#101层ResNet，和50层比，即将4个Blocks的units数量从3,4,6,3提升到3,4,23,3
def resnet_v2_101(inputs, num_classes = None, global_pool = True, reuse = None, scope = 'resnet_v2_50'):
	blocks = [Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]), 
			  Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
			  Block('block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
			  Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
	return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block = True, reuse = reuse, scope = scope)

#152层ResNet，将第二个Block的units数量提高到8，将第三个units数量提高到36
def resnet_v2_152(inputs, num_classes = None, global_pool = True, reuse = None, scope = 'resnet_v2_50'):
	blocks = [Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]), 
			  Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
			  Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
			  Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
	return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block = True, reuse = reuse, scope = scope)

#200层ResNet，将第二个Block的units数量提高到24，第三个units数量提高到36
def resnet_v2_200(inputs, num_classes = None, global_pool = True, reuse = None, scope = 'resnet_v2_50'):
	blocks = [Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]), 
			  Block('block2', bottleneck, [(512, 128, 1)] * 23+ [(512, 128, 2)]),
			  Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
			  Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
	return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block = True, reuse = reuse, scope = scope)



#===========性能评测=================================
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
height, width = 224, 224
inputs = tf.random_uniform((batch_size, height, width, 3))
with slim.arg_scope(resnet_arg_scope(is_training = False)):
	net, end_points = resnet_v2_152(inputs, 1000)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches = 100
time_tensorflow_run(sess, net, "Forward")