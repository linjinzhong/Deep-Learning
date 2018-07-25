#-*- encoding:utf-8 -*-

import numpy as np
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

def add_layer(inputs, in_size, out_size, activation_function=None):
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size]) + 0.01)

    Z = tf.matmul(inputs, W) + b 
    if activation_function is None:
        outputs = Z 
    else:
        outputs = activation_function(Z)

    return outputs


if __name__ == "__main__":

    MNIST = input_data.read_data_sets("MNIST/", one_hot=True)

    learning_rate = 0.01
    batch_size = 128
    n_epochs = 70

    X = tf.placeholder(tf.float32, [batch_size, 784])
    Y = tf.placeholder(tf.float32, [batch_size, 10])

    layer_dims = [784, 500, 500, 10]
    layer_count = len(layer_dims) - 1
    layer_iter = X 

    for l in range(1, layer_count):
        layer_iter = add_layer(layer_iter, layer_dims[l-1], layer_dims[l], activation_function=tf.nn.relu)
    prediction = add_layer(layer_iter, layer_dims[layer_count-1], layer_dims[layer_count], activation_function=None)

    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=prediction)
    loss = tf.reduce_mean(entropy)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        n_batches = int(MNIST.train.num_examples/batch_size)

        for i in range(n_epochs):
            for j in range(n_batches):
                X_batch, Y_batch = MNIST.train.next_batch(batch_size)
                _, loss_ = sess.run([optimizer, loss], feed_dict={X:X_batch, Y:Y_batch})
                if i % 10 == 5 and j == 0:
                    print("Loss of epochs[{0}]: {1}".format(i, loss_))

        # test
        n_batches = int(MNIST.test.num_examples/batch_size)
        total_correct_preds = 0
        for i in range(n_batches):
            X_batch, Y_batch = MNIST.test.next_batch(batch_size)
            preds = sess.run(prediction, feed_dict={X: X_batch, Y:Y_batch})
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))

            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

            total_correct_preds += sess.run(accuracy)

        print("Accuracy {0}".format(total_correct_preds/MNIST.test.num_examples))