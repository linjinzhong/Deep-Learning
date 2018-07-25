# -*- conding: utf-8 -*-
# 多层感知机(MLP)
# 没有隐含层的神经网络是线性的，解决不了XOR(异或)问题
# 理论上只要隐含结点足够多， 即使只有一个隐含层的神经网络也能拟合任意函数。
# 同时隐含层越多，越容易拟合复杂的函数。
# 研究表明： 为了拟合复杂函数需要的隐含结点的数目，基本上随着隐含层的数量增多呈现指数下降趋势。
# 层数越深，概念越抽象。
# 深层带来问题： 过拟合(Dropout)，参数难调试(Adagrad, Adam)，梯度弥散(ReLU)

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 
mnist = input_data.read_data_sets('MNIST/', one_hot=True)

sess = tf.InteractiveSession()

in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev = 0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
y_ = tf. placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1, W2) + b2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis = 1))

train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

tf.global_variables_initializer().run()


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
    if i % 100 == 0:
        print('batch:', i, ' test accuracy: ', accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


print('test accuracy: ', accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

