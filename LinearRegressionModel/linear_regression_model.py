# !/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np

class linearRegressionModel:
    
    def __init__(self, x_dimen):
        self.x_dimen = x_dimen
        self._index_in_epoch = 0
        self.constructModel()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    # 权重初始化为截断正太分布
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape=shape, mean=0.0, stddev = 0.1)
        return tf.Variable(initial)
    
    # 偏差初始化
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_datas:
            perm = np.arange(self._num_datas)
            np.random.shuffle(perm)
            self._datas = self._datas[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_datas
        end = self._index_in_epoch
        return self._datas[start:end],self._labels[start:end]

    # 线性回归模型构建
    def constructModel(self):
        self.x = tf.placeholder(tf.float32, [None, self.x_dimen])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.w = self.weight_variable([self.x_dimen, 1])
        self.b = self.bias_variable([1])
        self.y_prec = tf.nn.bias_add(tf.matmul(self.x,self.w), self.b)
        mse = tf.reduce_mean(tf.squared_difference(self.y_prec, self.y))
        l2 = tf.reduce_mean(tf.square(self.w))
        self.loss = mse + 0.15*l2
        self.train_step = tf.train.AdamOptimizer(0.1).minimize(self.loss)
    

    def train(self,x_train,y_train,x_test,y_test):
        self._datas = x_train
        self._labels = y_train
        self._num_datas = x_train.shape[0]
        for i in range(5000):
            batch = self.next_batch(100)
            self.sess.run(self.train_step,feed_dict={self.x:batch[0], self.y:batch[1]})
            if i%10 == 0:
                train_loss = self.sess.run(self.loss, feed_dict={self.x:batch[0], self.y:batch[1]})
                print('step %d,train_loss %f' % (i, train_loss))

    def predict_batch(self,arr,batch_size):
        for i in range(0,len(arr),batch_size):
            yield arr[i:i + batch_size]
            
    def predict(self, x_predict):
        pred_list = []
        for x_test_batch in self.predict_batch(x_predict,100):
            pred = self.sess.run(self.y_prec, {self.x:x_test_batch})
            pred_list.append(pred)
        return np.vstack(pred_list)