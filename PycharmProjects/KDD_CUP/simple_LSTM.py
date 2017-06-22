#coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import *


class Config(object):

    HIDDEN_SIZE = 12           # 隐藏层规模（cell的个数）
    INPUT_SIZE = 3              # 输入神经元的个数
    OUTPUT_SIZE = 1             # 输出神经元的个数
    NUM_LAYERS = 1              # 深层循环神经网络中LSTM的层数
    LEARNING_RATE = 0.2        # 学习速率
    NUM_STEPS = 6              # 训练数据的截断长度
    TRAIN_BATCH_SIZE = 32      # 训练集数据的batch大小
    TEST_BATCH_SIZE = 14       # 测试集数据的batch大小
    NUM_EPOCH = 200             # 在训练集上迭代的次数
    KEEP_PROB = 0.8             # 不被drop_out的概率
    SPLIT_RATE = 0.2            # 测试集所占的比例



class Simple_LSTM(object):
    def __init__(self, config, is_training):
        self.config = config
        # Parameter initialization
        if is_training:
            self.batch_size = self.config.TRAIN_BATCH_SIZE
        else:
            self.batch_size = self.config.TEST_BATCH_SIZE
        self.num_steps = self.config.NUM_STEPS
        self.input_size = self.config.INPUT_SIZE
        self.hidden_size = self.config.HIDDEN_SIZE
        self.output_size = self.config.OUTPUT_SIZE
        self.num_layers = self.config.NUM_LAYERS
        self.num_epoches = self.config.NUM_EPOCH
        self.keep_prob = self.config.KEEP_PROB
        self.learning_rate = self.config.LEARNING_RATE
        self.is_training = is_training
        self.split_rate = self.config.SPLIT_RATE

        with tf.name_scope('input'):
            # 定义输入层
            self.input_data = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_steps, self.input_size])
            # 定义预期输出
            self.targets = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_steps, self.output_size])

        with tf.variable_scope('in_hidden'):
            self.add_input_layer()

        with tf.variable_scope('cell'):
            self.add_cell()

        with tf.variable_scope('out_hidden'):
            self.add_output_layer()

        with tf.name_scope('cost'):
            self.compute_cost()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)



    def add_input_layer(self):
        input = tf.reshape(self.input_data, shape=[-1, self.input_size])
        Weights = self._weight_variable([self.input_size, self.hidden_size], name='weights')
        bias = self._bias_variable([self.hidden_size], name='bias')
        self.l_in_cell = tf.nn.relu6(tf.matmul(input, Weights) + bias)
        self.l_in_cell = tf.reshape(self.l_in_cell, [self.batch_size, self.num_steps, self.hidden_size])



    def add_cell(self):
        # 定义深层LSTM
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
        if self.is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = self.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.config.NUM_LAYERS)
        # 初始化最初的状态,全零向量
        initial_state = cell.zero_state(self.batch_size, tf.float32)
        output, state = tf.nn.dynamic_rnn(cell, self.l_in_cell, initial_state=initial_state, time_major=False)
        # shape(output) = (batch_size*num_steps, cell_size)
        self.cell_outputs = tf.reshape(output, shape=[-1,self.hidden_size])


    def add_output_layer(self):
        input = self.cell_outputs
        weights = self._weight_variable([self.hidden_size, self.output_size], name='weights')
        bias = self._bias_variable([self.output_size], name='bias')
        # shape(pred) = (batch_size*num_steps, output_size)
        self.pred = tf.nn.sigmoid(tf.matmul(input, weights) + bias)


    # def compute_cost(self):
    #     losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
    #         [tf.reshape(self.pred, [-1], name='reshape_pred')],
    #         [tf.reshape(self.targets, [-1], name='reshape_target')],
    #         [tf.ones([self.batch_size * self.num_steps * self.output_size], dtype=tf.float32)],
    #         average_across_timesteps=True,
    #         softmax_loss_function=self.ms_error,
    #         name='losses'
    #     )
    #     self.cost = tf.div(tf.reduce_sum(losses, name='losses_sum'),self.batch_size, name='average_cost')
    #     tf.summary.scalar('cost', self.cost)
    #
    # def ms_error(self, y_pre, y_target):
    #     return tf.square(y_pre - y_target)


    # 计算costs时,若使用MAPE作为损失, 可能会出现分母为0的情况, cost = nan, 模型无法训练, 所以改用MAE近似代替MAPE
    def compute_cost(self):
        prediction = tf.reshape(self.pred, [self.batch_size, self.num_steps, self.output_size])
        targets = tf.reshape(self.targets, [self.batch_size, self.num_steps, self.output_size])
        self.cost = tf.reduce_mean(tf.abs(targets - prediction))



    def _weight_variable(self, shape, name='weights'):
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=2.0)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)



    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)




# 使用给定的模型在数据data上运行一次
def run_epoch(sess, model, train_feature, test_feature, train_label, test_label, data_max, data_min):
    total_costs = []
    batch_generator = batch_reader(train_feature, test_feature, train_label, test_label, batch_size = model.batch_size,is_training=model.is_training)
    batch_generator.reset()                                                                      # 数据重置
    for i_batch in range(batch_generator.num_batches):
        # shape(feature_batch) = (128, 18) ,shape(label_batch) = (128, 6)
        feature_batch, label_batch = batch_generator.next_batch()

        feed_dict = {
            model.input_data: feature_batch.reshape([model.batch_size, model.num_steps, model.input_size]),
            model.targets: label_batch.reshape([model.batch_size, model.num_steps, model.output_size])
        }

        # 如果模型处于训练状态,则对模型进行优化
        if model.is_training:
            # shape(prediction) = [batch_size * num_steps, output_size]
            _, cost, prediction = sess.run([model.train_op, model.cost, model.pred], feed_dict = feed_dict)

            prediction = np.reshape(prediction,[-1, model.output_size])
            targets = np.reshape(label_batch, [-1, model.output_size])
            pratical_targets = liner_denormalization(targets, data_max, data_min)             #去归一化
            pratical_prediction = liner_denormalization(prediction, data_max, data_min)     #去归一化
            cost = cal_MAPE(pratical_targets, pratical_prediction)                               #去归一化之后的MAE

        # 如果模型处于测试状态,则不优化
        else:
            # shape(prediction) = [batch_size * num_steps, output_size]
            # 在真实数据集上计算损失,需要将数据去归一化
            _, cost, prediction = sess.run([model.train_op, model.cost, model.pred], feed_dict = feed_dict)
            prediction = np.reshape(prediction,[-1, model.output_size])
            targets = np.reshape(label_batch, [-1, model.output_size])
            pratical_targets = liner_denormalization(targets, data_max, data_min)             #去归一化
            pratical_prediction = liner_denormalization(prediction, data_max, data_min)     #去归一化
            cost = cal_MAPE(pratical_targets, pratical_prediction)                                                      #得到在真实值上的MAE
        total_costs.append(cost)

    print "costs:%.4f" % np.mean(total_costs)


# 在测试数据上运行,得到用于提交的预测值
def test_run_epoch(sess, model, a, data_max, data_min):
    test_feature = load_route_test_feature(a)

    test_feature = np.reshape(test_feature, [model.batch_size, model.num_steps, model.input_size])
    feed_dict = {
        model.input_data: test_feature
    }
    prediction = sess.run([model.pred], feed_dict = feed_dict)                       # shape(prediction) = (14,6,1)
    prediction = np.reshape(prediction, [-1, model.output_size])                     # shape(prediction) = (84,)
    pratical_prediction = liner_denormalization(prediction, data_max, data_min)      # 去归一化
    pratical_prediction = np.reshape(pratical_prediction, [-1, ])
    pd.Series(pratical_prediction).to_csv('/home/johnson/PycharmProjects/KDD_CUP/average_time_results/result_by_route/best_B_1.csv')
    print "predict has finished!"




if __name__ == '__main__':
    # 参数a指定路径
    a = 2
    train_feature, test_feature, train_label, test_label = load_route_feature_label(a)
    # shape(data_max) = (6,), shape(data_min) = (6,)
    normalized_train_data, normalized_test_data, data_max, data_min = get_liner_normalizer_data()
    config = Config()


    # 定义训练时使用的神经网络模型
    with tf.variable_scope("model",reuse=None):
        train_model = Simple_LSTM(config, is_training=True)

    # # 定义测试所用的循环神经网络模型
    # with tf.variable_scope("model",reuse=True):
    #     test_model = Simple_LSTM(config, is_training=False)

    #得到最终预测结果的模型
    with tf.variable_scope("model",reuse=True):
        vaild_model = Simple_LSTM(config, is_training=False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i_epoch in range(train_model.num_epoches):
            print "In iteration: %d, training costs is:" % (i_epoch + 1),       # 加逗号,使输出不换行
            # 使用训练数据训练模型
            run_epoch(sess, model = train_model, train_feature = train_feature, test_feature = test_feature, train_label = train_label, test_label = test_label, data_max = data_max[a], data_min = data_min[a])
        print "The process of training has finished!"

        # # 使用本地验证数据集验证模型效果
        # print "MAPE on testset",
        # run_epoch(sess, model = test_model, train_feature = train_feature, test_feature = test_feature, train_label = train_label, test_label = test_label, data_max = data_max[a], data_min = data_min[a])


        #得到最终预测结果
        test_run_epoch(sess, model=vaild_model, a=a, data_max = data_max[a], data_min = data_min[a])




















