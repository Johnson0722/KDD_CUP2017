# coding:utf-8
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from utils import *

# 每条路径单独预测, 输入的参数a为整型, 代表第几条路径
def load_train_test_data(a):
    # train data
    nor_train_traffic_data = load_train_traffic()                                   # shape = (6552, 6)
    nor_train_time_data, _, data_max, data_min = get_liner_normalizer_data()        # shape = (6556, 6)
    train_input_data = np.stack([nor_train_time_data[:,a], nor_train_traffic_data[:,a]], axis=1) # shape = (6556, 2)
    train_output_data = load_train_time()[:,a]                                      # shape = (6556,1)

    # test data
    nor_test_traffic_data = load_test_traffic()                                                 # shape = (84, 6)
    _, nor_test_time_data, data_max, data_min = get_liner_normalizer_data()                     # shape = (84,6)
    test_input_data = np.stack([nor_test_time_data[:,a], nor_test_traffic_data[:,a]], axis=1)   # shape = (84,2)

    return train_input_data,train_output_data, test_input_data


# 生成训练集的特征和标签
def generate_train_feature_label(input_data, output_data, holiday_data = None, time_data = None):
    num_steps = 6
    data_length = np.shape(input_data)[0]                     # 数据集的长度
    input_size = 2
    other_feature_size = 0
    output_size = 1
    data_length_index = np.arange(data_length)
    data_feature = np.zeros((data_length - 2*num_steps, num_steps*input_size + other_feature_size))           # 初始化训练集（输入）
    data_label = np.zeros((data_length - 2*num_steps, num_steps*output_size))                                 # 初始化训练集（预测值）

    for i in range(data_length - 2*num_steps):
        feature1 = np.reshape(input_data[:,0][data_length_index[i:i + num_steps]], [-1,])                             # 前两小时的平均时间
        feature2 = np.reshape(input_data[:,1][data_length_index[i:i + num_steps]], [-1,])                             # 前两小时的车流了特征
        feature3 = holiday_data[i + num_steps]                                                                        # 加入节假日特征
        data_feature[i] = np.concatenate([feature1, feature2], axis=0)                                         # 总特征
        data_label[i] = np.reshape(output_data[data_length_index[i + num_steps: i + 2*num_steps]], [-1, ])

    # shape(train_feature) = (6540, 12), shape(train_label) = (6540, 6)
    return data_feature, data_label


def morning_afternoon_index():
    morning_index_array = np.arange(15,24)  # 一天里属于白天的索引
    afternoon_index_array = np.arange(45,57)
    morning_index = list(morning_index_array)
    afternoon_index = list(afternoon_index_array)
    total_days = 90                       # 从7月19号到10月17号总天数
    for i in range(total_days):
        morning_index_array += 72
        afternoon_index_array += 72
        morning_index.extend(list(morning_index_array))
        afternoon_index.extend(list(afternoon_index_array))
    # len(morning_index) = 91, len(afternoon_index = 91)
    return morning_index, afternoon_index



def generate_test_feature(test_input_data, holiday_data = None):
    feature1 = np.reshape(test_input_data[:,0], [14, -1])                                                          # 前两小时的平均时间特征
    feature2 = np.reshape(test_input_data[:,1], [14, -1])                                                          #  前两小时的车流了特征
    feature3 = holiday_data                                                                                        # 节假日特征 shape = (14,2)
    test_feature = np.concatenate([feature1, feature2], axis=1)
    return test_feature                                                                                            # shape = (14,14)



def predict_by_route(a):
    morning_index, afternoon_index = morning_afternoon_index()
    train_holiday_data = load_train_holidays()                                                                          # shape = (6552, 2)
    test_holiday_data = load_test_holidays()
    train_input_data,train_output_data, test_input_data = load_train_test_data(a)                                       # A_2的数据
    data_feature, data_label = generate_train_feature_label(train_input_data, train_output_data, holiday_data=train_holiday_data)

    # shape(day_feature) = (3276, 12), shape(day_label) = (3276,6), 索引出上午的数据和下午的数据
    morning_feature, morning_label = data_feature[morning_index], data_label[morning_index]
    afternoon_feature, afternoon_label = data_feature[afternoon_index], data_label[afternoon_index]

    # 生成本地训练集和测试集
    train_feature, test_feature, train_label, test_label = train_test_split(afternoon_feature, afternoon_label, test_size=0.2, random_state=15)

    clf = linear_model.LinearRegression()
    clf.fit(train_feature, train_label)
    print "linear regression model's MAE is", cal_MAPE(test_label, clf.predict(test_feature))

    # test_feature = generate_test_feature(test_input_data, holiday_data=test_holiday_data)
    # predict = np.reshape(lasso.predict(test_feature), [-1,])                               # shape = (84,)
    # return predict


if __name__ == '__main__':

    predict_A_2 = predict_by_route(0)
    predict_A_3 = predict_by_route(1)
    predict_B_1 = predict_by_route(2)
    predict_B_3 = predict_by_route(3)
    predict_C_1 = predict_by_route(4)
    predict_C_3 = predict_by_route(5)
    # predict_value = np.stack([predict_A_2, predict_A_3, predict_B_1, predict_B_3, predict_C_1, predict_C_3], axis=1)        # shape = (84, 6)
    # result_DataFrame = generate_submitted_result(predict_value)
    # result_DataFrame.to_csv('average_time_results/by_route_linear_regression.csv', index=False, columns=['intersection_id','tollgate_id','time_window','avg_travel_time'])
