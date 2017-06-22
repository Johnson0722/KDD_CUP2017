# coding:utf-8
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from utils import *

def load_train_data():
    train_traffic_data = load_train_traffic()                                                             # shape(time) = shape(traffic) = (6552, 6)
    train_time_data, _, data_max, data_min = get_liner_normalizer_data()                                  # shape(timne_data) = (6556, 6)
    total_input_data = np.concatenate([train_time_data, train_traffic_data], axis=1)                      # shape(total_input_data) = (6556,12)
    output_data = load_train_time()                                                                       # shape(output_data) = (6552, 6), without_normalization
    return total_input_data, output_data



def load_test_data():
    _, test_time_data, data_max, data_min = get_liner_normalizer_data()                                 # shape(test_data) = (84,6)                                                        # shape(test_weather_data) = (84,4)
    test_traffic_data = load_test_traffic()                                                             # shape(time) = shape(traffic) = (84, 6)
    test_input_data = np.concatenate([test_time_data, test_traffic_data], axis=1)                       # shape(test_input_data) = (84,12)

    return test_input_data




def generate_feature_label_data(input_data, output_data, split_rate):
    num_steps = 6
    data_length = np.shape(input_data)[0]                     # 数据集的长度
    input_size = np.shape(input_data)[1]
    output_size = np.shape(output_data)[1]
    test_length = int(data_length * split_rate)               # 测试集的长度
    train_length = data_length - test_length                  # 训练集的长度
    train_length_index = np.arange(train_length)
    test_length_index = np.arange(test_length)
    train_data = np.zeros((train_length - 2*num_steps, num_steps*input_size))                  # 初始化训练集（输入）
    train_label = np.zeros((train_length - 2*num_steps, num_steps*output_size))                # 初始化训练集（预测值）
    test_data = np.zeros((test_length - 2*num_steps, num_steps*input_size))                    # 初始化测试集（输入）
    test_label = np.zeros((test_length - 2*num_steps, num_steps*output_size))                  # 初始化测试集（预测值）

    for i in range(train_length - 2*num_steps):
        train_data[i] = np.reshape(input_data[train_length_index[i:i + num_steps]], [-1,])
        train_label[i] = np.reshape(output_data[train_length_index[i + num_steps: i + 2*num_steps]], [-1, ])

    for i in range(test_length - 2*num_steps):
        test_data[i] = np.reshape(input_data[train_length + test_length_index[i:i + num_steps]], [-1,])
        test_label[i] = np.reshape(output_data[train_length + test_length_index[i + num_steps: i + 2*num_steps]], [-1,])

    return train_data, train_label, test_data, test_label



total_input_data,  = load_train_data()   # 生成训练集
test_input_data = load_test_data()                  # 加载最终测试数据集


train_x, train_y, test_x, test_y = generate_feature_label_data(input_data=total_input_data, output_data=output_data, split_rate=0.1)
# print train_x.shape         # shape = (5885, 96)
# print train_y.shape         # shape = (5885, 36)
# print test_x.shape          # shape = (643, 96)
# print test_y.shape          # shape = (643, 36)

# clf = linear_model.LinearRegression(normalize=False)
# clf.fit(train_x, train_y)
#
# reg = linear_model.Ridge(alpha=10)
# reg.fit(train_x,train_y)

# lasso = linear_model.Lasso(alpha=0.1)
# lasso.fit(train_x, train_y)


# rng = linear_model.MultiTaskLasso(alpha=0.01)
# rng.fit(train_x, train_y)

# Ela = linear_model.ElasticNet(alpha=0.005, l1_ratio=0.9)
# Ela.fit(train_x, train_y)

# lar = linear_model.Lars()
# lar.fit(train_x, train_y)

# rf = ensemble.RandomForestRegressor()
# rf.fit(train_x, train_y)

# krr = KernelRidge()
# krr.fit(train_x, train_y)

# print "linear regression model's MAE is", mean_absolute_error(test_y, clf.predict(test_x))
# print "Ridge regression model's MAE is", mean_absolute_error(test_y, reg.predict(test_x))
# print "lasso regression model's MAE is", mean_absolute_error(test_y, lasso.predict(test_x))   #42.029
# print "MultiTasklasso regression model's MAE is", mean_absolute_error(test_y, rng.predict(test_x))
# print "ElasticNet regression model's MAE is", mean_absolute_error(test_y, Ela.predict(test_x))
# print "LARS regression model's MAE is", mean_absolute_error(test_y, lar.predict(test_x))
# print "random forest model's MAE is", mean_absolute_error(test_y, rf.predict(test_x))
# print rf.feature_importances_

# print "kernel ridge regression model's MAE is", mean_absolute_error(test_y, krr.predict(test_x))


# pred = lasso.predict(test_input_data)   # shape = (14, 36)
#
# pred = np.reshape(pred, [-1, 6])
# lasso_results = generate_submitted_result(pred)
# lasso_results.to_csv('average_time_results/lasso_results_exception_value.csv', index=False, columns=['intersection_id','tollgate_id','time_window','avg_travel_time'])

