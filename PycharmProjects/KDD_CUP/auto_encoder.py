# coding: utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
from utils import get_liner_normalizer_data


# Autoencoder definition
def autoencoder(dimensions=[6, 3, 1]):
    """Build a deep autoencoder w/ tied weights.
    Parameters
    ----------
    dimensions : list, optional
        The number of neurons for each layer of the autoencoder.
    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    """
    # %% input to the network
    keep_prob = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
    current_input = tf.nn.dropout(x, keep_prob=keep_prob)

    # %% Build the encoder
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(tf.zeros([n_input, n_output]))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
        current_input = output

    # %% latent representation
    z = current_input   ##compressed representation
    encoder.reverse()

    # %% Build the decoder using the same weights
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
        current_input = output

    # %% now have the reconstruction through the network
    y = current_input

    # %% cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x))
    #cost = tf.nn.sigmoid_cross_entropy_with_logits(y,x)
    #cost = tf.reduce_mean(-tf.reduce_sum(x * tf.log(y),reduction_indices=[1]))

    return {'x': x, 'z': z, 'y': y, 'corrupt_prob':keep_prob,'cost': cost}


# 从路径a的周围路段中提取特征
def get_load_features():
    # shape(normalized_train_data) = (6552,6), shape(normalized_test_data) = (normalized_test_data) = (84,6)
    normalized_train_data, normalized_test_data, data_max, data_min = get_liner_normalizer_data()
    All_normalized_data = np.concatenate([normalized_train_data, normalized_test_data], axis=0) # shape = (6536, 6)
    ae = autoencoder(dimensions=[6, 3, 1])                                              ##dimension[0] = number of cells
    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(ae['cost'])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_size = 128
    n_epochs = 40
    num_batches = np.shape(All_normalized_data)[0]/batch_size

    for epoch_i in range(n_epochs):
        for batch_i in range(num_batches):
            trainSet = All_normalized_data[batch_i*batch_size:(batch_i+1)*batch_size,:]
            sess.run(optimizer,feed_dict={ae['x']:trainSet,ae['corrupt_prob']: [0.8]})
    mat_compressed =  sess.run(ae['z'],feed_dict={ae['x']:All_normalized_data,ae['corrupt_prob']: [0.8]}) ##shape(mat_compressed) = (6536,1)
    train_load_features = mat_compressed[:6552]
    test_load_features = mat_compressed[6552:]
    # shape(train_load_features) = (6552, 1), shape(test_load_features) = (84,1)
    return train_load_features, test_load_features

# a,b = get_load_features()
#
# print a.shape
# print b.shape
# print b