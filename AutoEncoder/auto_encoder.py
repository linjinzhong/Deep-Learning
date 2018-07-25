# -*- coding:utf-8 -*-
# 自编码器：
# 特征可以不断抽象转为高一级的特征。
# 特征的稀疏表达使用少量的基本特征组合拼装得到更高层抽象的特征。
# 目标：使用稀疏的高阶特征重新组合来重构自己。
# 期望输入输出一致。希望用高阶特征重构自己而不是简单的复制像素点。
# 可以用来逐层训练提取特征，将网络初始化到一个比较好的位置，辅助后面的监督训练。
# 学习数据频繁出现的 模式 和 结构。

import numpy as np 
import sklearn.preprocessing as prep 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

# 参数初始化方法：xavier initialization
# 根据某一层网络的输入输出结点数量自动调整最合适的分布
# 如果深度学习模型初始化权重太小，那信号将会在每层间传递时逐渐缩小而难以产生作用；
# 如果初始化太大，那么信号将在每层间传递时逐渐放大并导致发散和实效
# xavier 让权重满足0均值同时方差为2/(N_in+N_out)
# 分布可以用均匀分布或者高斯分布
# 均匀分布：创建一个(-sqrt(6/(N_in+N_out)), -sqrt(6/(N_in+N_out)))均匀分布
# 根据均匀分布方差公式(max-min)^2/12,刚好满足要求
def xavier_init(n_in, n_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (n_in + n_out))
    high = constant * np.sqrt(6.0 / (n_in + n_out))
    return tf.random_uniform((n_in, n_out), minval = low, maxval = high, dtype = tf.float32)



# 去噪自编码器
# 激活函数： softplus=ln(1+e^x)
#           relu = max(0, x)
class AdditiveGaussianNoiseAutoEncoder(object):
    def __init__(self, n_input, n_hidden, activate_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.activate = activate_function
        self.scale = tf.placeholder(tf.float32) 
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # 隐藏层，输出层
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden  = self.activate(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # 损失函数
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        # self.cost = tf.reduce_mean(tf.squared_difference(self.reconstruction, self.x))

        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    # 初始化权重
    # 由于输出层没有激活函数，所以可以初始化为w2,b2为0
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    # 训练的一部分
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    # 只计算损失，用于测试部分
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    # transform函数，返回自编码器隐含层输出结果，目的是提供一个接口来获取抽象后的特征
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: training_scale})

    # 将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据。
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})

    # 整体运行复原过程： 提取高阶特征(transform) 和 通过高阶特征复原数据(generate)
    def reconstrut(self, X):
        return self.sess.run(self.reconstruction, feed_dict = {self.x: X, self.scale: training_scale})

    def getWeight(self):
        return self.sess.run(self.weights['w1'])
    def getBiases(self):
        return self.sess.run(self.weights['b1'])


# 标准化处理数据，0均值1方差
# 现在训练集行fit，再将这个scaler用到训练数据和测试数据上
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

# 定义一个获取随机block数据的函数
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

if __name__ == '__main__':

    mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

    n_samples = int(mnist.train.num_examples)
    n_epochs = 20
    batch_size = 128
    display_step = 1

    autoencoder = AdditiveGaussianNoiseAutoEncoder(n_input = 784, n_hidden = 200, activate_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(learning_rate = 0.001), scale = 0.01)

    for epoch in range(n_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)
            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    # 测试
    print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))