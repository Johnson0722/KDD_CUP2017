# coding:utf-8
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from utils import *
from auto_encoder import *

# 每条路径单独预测, 输入的参数a为整型, 代表第几条路径
def load_train_test_data(a):
    # train data
    nor_train_traffic_data = load_train_traffic()                               # shape = (6552, 6)
    nor_train_time_data, _, data_max, data_min = get_liner_normalizer_data()    # shape = (6556, 6)
    train_input_data = np.stack([nor_train_time_data[:,a], nor_train_traffic_data[:,a]], axis=1) # shape = (6556, 2)
    train_output_data = load_train_time()[:,a]                                  # shape = (6556, 1)

    # test data
    nor_test_traffic_data = load_test_traffic()                                                 # shape = (84, 6)
    _, nor_test_time_data, data_max, data_min = get_liner_normalizer_data()                     # shape = (84,6)
    test_input_data = np.stack([nor_test_time_data[:,a], nor_test_traffic_data[:,a]], axis=1)   # shape = (84,2)

    return train_input_data,train_output_data, test_input_data


# 生成训练集的特征和标签, 输入input_data为指定路径的平均时间和车流量信息, output_data为指定路径的平均时间信息
def generate_train_feature_label(input_data, output_data, holiday_data = None, weather_data = None):
    num_steps = 6
    data_length = np.shape(input_data)[0]                     # 数据集的长度
    input_size = 3
    other_feature_size = 2
    output_size = 1
    train_load_features, test_load_features = get_load_features()   # shape(train_load_features) = (6552, 1)
    data_length_index = np.arange(data_length)
    data_feature = np.zeros((data_length - 2*num_steps, num_steps*input_size + other_feature_size))            # 初始化训练集（输入）
    data_label = np.zeros((data_length - 2*num_steps , num_steps*output_size))                                 # 初始化训练集（预测值）
    for i in range(data_length - 2*num_steps):
        feature1 = np.reshape(input_data[:,0][data_length_index[i:i + num_steps]], [-1,])                             # 前两小时的平均时间
        feature2 = np.reshape(input_data[:,1][data_length_index[i:i + num_steps]], [-1,])                             # 前两小时的车流了特征
        feature3 = holiday_data[i + num_steps]                                                                        # 加入节假日特征
        feature4 = np.array(weather_data[i + num_steps*2, 2:])                                                        # 加入温度和湿度的特征
        feature5 = np.reshape(input_data[:,0][data_length_index[i - 504 + num_steps :i - 504 + 2*num_steps]], [-1,])  # 加入上周8点到10点特征
        feature6 = np.reshape(train_load_features[data_length_index[i:i + num_steps]], [-1,])                         # 加载道路特征信息
        data_feature[i] = np.concatenate([feature1, feature2, feature4, feature6], axis=0)                                 # 总特征
        data_label[i] = np.reshape(output_data[data_length_index[i + num_steps: i + 2*num_steps]], [-1, ])
    # shape(data_feature) = (6540, 20), shape(data_label) = (6540, 6)
    return data_feature, data_label


# 去除国庆节, 生成没有国庆节的索引
def filter_holiday_index():
    holiday_index = []
    holiday_start = 74
    holidat_stop = 81
    for i in range(holiday_start, holidat_stop):
        holiday_index.extend(list(range(i*72,(i+1)*72)))
    total_index = range(6540)
    No_holiday_index = list(set(total_index) - set(holiday_index))
    return No_holiday_index


# 生成最终提交测试集的特征, 输入input为测试的时间数据, holiday_data和weather_data均为测试集的节假日信息和温度信息
def generate_test_feature(test_input_data, holiday_data = None, weather_data = None):                              # shape(weather_data) = (84,4)
    train_load_features, test_load_features = get_load_features()                                                  # shape(test_load_features) = (84, 1)
    feature1 = np.reshape(test_input_data[:,0], [14, -1])                                                          # 前两小时的平均时间特征
    feature2 = np.reshape(test_input_data[:,1], [14, -1])                                                          #  前两小时的车流了特征
    feature3 = holiday_data                                                                                        # 节假日特征 shape = (14,2)
    feature4 = np.array(weather_data[::6, 2:])                                                                     # 加入温度和湿度特征
    feature6 = np.reshape(test_load_features[:,0], [14, -1])                                                       # 加入道路信息特征,shape = (14, 6)
    test_feature = np.concatenate([feature1, feature2, feature4, feature6], axis=1)
    return test_feature                                                                                            # shape = (14,14)


# 针对每条路径单独预测
def predict_by_route(a):
    No_holiday_index = filter_holiday_index()
    train_holiday_data = load_train_holidays()                                              # shape = (6552, 2)
    test_holiday_data = load_test_holidays()                                                # shape = (84, 2)
    normalized_train_weather_data, normalized_test_weather_data = load_weather_data()
    train_input_data,train_output_data, test_input_data = load_train_test_data(a)           # shape(train_input_data) = (6552, 2)
    # 训练数据上的特征和标签                                                                   # shape(train_output_data) = (6552, 1)
    data_feature, data_label = generate_train_feature_label(train_input_data, train_output_data, holiday_data=train_holiday_data, weather_data=normalized_train_weather_data)
    # 过滤掉国庆节
    data_feature = data_feature[No_holiday_index]
    data_label = data_label[No_holiday_index]
    # 生成本地训练集和测试集
    train_feature, test_feature, train_label, test_label = train_test_split(data_feature, data_label, test_size=0, random_state=10)

    # 线性模型
    clf = linear_model.LinearRegression()
    clf.fit(train_feature, train_label)
    # print "All time linear's MAE is", cal_MAPE(test_label, clf.predict(test_feature))

    test_feature = generate_test_feature(test_input_data, holiday_data=test_holiday_data, weather_data=normalized_test_weather_data)    # shape = (14,14)
    predict = np.reshape(clf.predict(test_feature), [-1,])                                   # shape = (84,)
    return predict




def get_all_time_results():
    all_time_predict_A_2 = predict_by_route(0)
    all_time_predict_A_3 = predict_by_route(1)
    all_time_predict_B_1 = predict_by_route(2)
    all_time_predict_B_3 = predict_by_route(3)
    all_time_predict_C_1 = predict_by_route(4)
    all_time_predict_C_3 = predict_by_route(5)
    return all_time_predict_A_2, all_time_predict_A_3, all_time_predict_B_1, all_time_predict_B_3, all_time_predict_C_1, all_time_predict_C_3


