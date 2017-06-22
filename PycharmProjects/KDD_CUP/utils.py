#coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
import tensorflow as tf
from sklearn.model_selection import train_test_split




# 加载训练数据集, 没有归一化
def load_train_time():
    train_time_array, normalized_train_traffic_array, test_time_array, normalized_test_traffic_array = Add_congestion_traffic()
    return train_time_array                         # shape = (6552,6)


# 加载测试数据集(for submitting), 没有归一化
def load_test_time():
    train_time_array, normalized_train_traffic_array, test_time_array, normalized_test_traffic_array = Add_congestion_traffic()
    return test_time_array                          # shape = (84,6)


# 加载交通流量训练集, 归一化之后
def load_train_traffic():
    train_time_array, normalized_train_traffic_array, test_time_array, normalized_test_traffic_array = Add_congestion_traffic()
    return normalized_train_traffic_array        # shape = (6552,6)



# 加载交通流量测试集, 归一化之后
def load_test_traffic():
    train_time_array, normalized_train_traffic_array, test_time_array, normalized_test_traffic_array = Add_congestion_traffic()
    return normalized_test_traffic_array        # shape = (84,6)



# 提取节假日特征, 并将得到的结果写入文件
def generate_holidays():
    date = pd.date_range(start = '7/19/2016 00:00:00', end = '10/17/2016 23:40:00', freq = '20min')
    # 生成工作日的信息, 对于中秋节和国庆节单独考虑
    Bussiness_day = pd.date_range(start = '7/19/2016 00:00:00', end = '10/17/2016 23:40:00', freq='B')
    Bussiness_day = list(Bussiness_day)
    Bussiness_day.remove(pd.Timestamp('2016-09-15 00:00:00', freq='B'))
    Bussiness_day.remove(pd.Timestamp('2016-09-16 00:00:00', freq='B'))
    Bussiness_day.remove(pd.Timestamp('2016-10-03 00:00:00', freq='B'))
    Bussiness_day.remove(pd.Timestamp('2016-10-04 00:00:00', freq='B'))
    Bussiness_day.remove(pd.Timestamp('2016-10-05 00:00:00', freq='B'))
    Bussiness_day.remove(pd.Timestamp('2016-10-06 00:00:00', freq='B'))
    Bussiness_day.remove(pd.Timestamp('2016-10-07 00:00:00', freq='B'))
    Bussiness_day.append(pd.Timestamp('2016-09-18 00:00:00', freq='B'))
    Bussiness_day.append(pd.Timestamp('2016-10-08 00:00:00', freq='B'))
    Bussiness_day.append(pd.Timestamp('2016-10-09 00:00:00', freq='B'))
    work_day = []                 # 生成一个列表存放节假日信息
    [work_day.append(str(day)[:10]) for day in Bussiness_day]

    # 初始化节假日信息, 节假日信息用one-hot来表达 [0,1]代表节假日,[1,0]代表工作日
    hoilday_df = pd.DataFrame(data = np.zeros([6552,2]), index=date)

    # 遍历所有日期填充节假日信息
    for i in range(len(hoilday_df)):
        # 提取出年月日信息即可
        if str(hoilday_df.index[i])[:10] in work_day:
            hoilday_df.ix[i] = [1,0]        # 工作日[1,0]
        else:
            hoilday_df.ix[i] = [0,1]        # 节假日[0,1]
    hoilday_df.to_csv('preprocessed_data/train_data/holiday_date.csv')
    # shape = (6552, 2)

# 加载训练时节假日数据, 返回一个二维矩阵
def load_train_holidays():
    holidays_df = pd.read_csv('preprocessed_data/train_data/holiday_date.csv', index_col=0)
    return holidays_df.values  #shape = (6552, 2)



# 加载测试时的节假日数据 (2016-10-18 - 2016-10-21, 2016-10-24工作日, 2016-10-22, 2016-10-23休息日)
def load_test_holidays():
    work_day_18_21 = np.reshape([1, 0]*8, [8, 2])
    holiday_22_23 = np.reshape([0, 1]*4, [4,2])
    work_day_23 = np.reshape([1, 0]*2, [2,2])
    test_holidays = np.concatenate([work_day_18_21, holiday_22_23, work_day_23], axis=0)
    return test_holidays        #shape = (14, 2)




# 加载指定路段的信息, PCA降维处理
# 输入time_data为某条路径的初始的时间序列
def add_load_features(time_data):          # shape(time_data) = (6552,)
    data_length = len(time_data)
    data_length_index = np.arange(data_length)
    train_samples = np.zeros([data_length - 12, 6])     # shape(train_samples) = (6552, 6)

    for i in range(data_length - 12):
        train_samples[i] = np.reshape(time_data[data_length_index[i:i + 6]], [-1,])     # 滑动得到训练样本

    pca = PCA(n_components=1)
    pca_features = pca.fit_transform(train_samples)
    return pca_features                      # shape = (6540, 1)


# 加载所有路段的特征信息
def add_all_load_features():
    normalized_train_data, normalized_test_data, data_max, data_min, = get_liner_normalizer_data()
    pca_features = np.zeros([6540, 6])
    for i in range(6):
        pca_features[:,i] = add_load_features(normalized_train_data[:,i])[:,0]
    return pca_features


# 加载天气数据(包括训练集和测试集),归一化之后
def load_weather_data():
    #加载训练天气数据
    weather_data = pd.read_csv('/home/johnson/PycharmProjects/KDD_CUP/preprocessed_data/train_data/weather_date.csv',index_col=0)
    train_weather_data =  weather_data.values                         # shape(weather_array) = (6552, 4)

    #加载测试训练数据
    test_weather_data = pd.read_csv('/home/johnson/PycharmProjects/KDD_CUP/preprocessed_data/test_data/test_weather_date.csv')
    # 筛选出所需的日期,8点到10点,5点到7点
    index = range(24,30) + range(51,57) + range(24+72, 30+72) + range(51+72, 57+72) + range(24+72*2, 30+72*2) + range(51+72*2, 57+72*2) + \
            range(24+72*3, 30+72*3) + range(51+72*3, 57+72*3) + range(24+72*4, 30+72*4) + range(51+72*4, 57+72*4) + \
            range(24+72*5, 30+72*5) + range(51+72*5, 57+72*5) + range(24+72*6, 30+72*6) + range(51+72*6, 57+72*6)
    test_weather_data = test_weather_data.reindex(index=index)
    test_weather_data = test_weather_data.values[:,1:]               # shape(test_weather_data) = (84,4)

    #数据归一化操作
    min_max_scalar = preprocessing.MinMaxScaler()
    total_weather_data = np.concatenate([train_weather_data, test_weather_data], axis=0)
    normalized_weather_array = min_max_scalar.fit_transform(total_weather_data)
    normalized_train_weather_array = normalized_weather_array[:6552]
    normalized_test_weather_array = normalized_weather_array[6552:]

    return normalized_train_weather_array, normalized_test_weather_array



# 加载数据集和,交通流量特征数据
def Add_congestion_traffic():
    # load train_time_flow
    train_time_flow_A_2 = pd.read_csv('preprocessed_data/train_data/time_flow_A_2.csv', index_col=0)
    train_time_flow_A_3 = pd.read_csv('preprocessed_data/train_data/time_flow_A_3.csv', index_col=0)
    train_time_flow_B_1 = pd.read_csv('preprocessed_data/train_data/time_flow_B_1.csv', index_col=0)
    train_time_flow_B_3 = pd.read_csv('preprocessed_data/train_data/time_flow_B_3.csv', index_col=0)
    train_time_flow_C_1 = pd.read_csv('preprocessed_data/train_data/time_flow_C_1.csv', index_col=0)
    train_time_flow_C_3 = pd.read_csv('preprocessed_data/train_data/time_flow_C_3.csv', index_col=0)
    #shape = (6552,6) type = np.array   without normalization
    train_time_array = pd.DataFrame([train_time_flow_A_2['travel_time'],train_time_flow_A_3['travel_time'],train_time_flow_B_1['travel_time'],
                               train_time_flow_B_3['travel_time'],train_time_flow_C_1['travel_time'],train_time_flow_C_3['travel_time']]).values.T
    train_traffic_array = pd.DataFrame([train_time_flow_A_2['traffic_flow'],train_time_flow_A_3['traffic_flow'],train_time_flow_B_1['traffic_flow'],
                                  train_time_flow_B_3['traffic_flow'],train_time_flow_C_1['traffic_flow'],train_time_flow_C_3['traffic_flow']]).values.T

    # load test time flow
    test_time_flow_A_2 = pd.read_csv('preprocessed_data/test_data/test_time_flow_A_2.csv', index_col=0)
    test_time_flow_A_3 = pd.read_csv('preprocessed_data/test_data/test_time_flow_A_3.csv', index_col=0)
    test_time_flow_B_1 = pd.read_csv('preprocessed_data/test_data/test_time_flow_B_1.csv', index_col=0)
    test_time_flow_B_3 = pd.read_csv('preprocessed_data/test_data/test_time_flow_B_3.csv', index_col=0)
    test_time_flow_C_1 = pd.read_csv('preprocessed_data/test_data/test_time_flow_C_1.csv', index_col=0)
    test_time_flow_C_3 = pd.read_csv('preprocessed_data/test_data/test_time_flow_C_3.csv', index_col=0)
    # shape = (84,6) type = np.array    without normalization
    test_time_array = pd.DataFrame([test_time_flow_A_2['travel_time'],test_time_flow_A_3['travel_time'],test_time_flow_B_1['travel_time'],
                                    test_time_flow_B_3['travel_time'],test_time_flow_C_1['travel_time'],test_time_flow_C_3['travel_time']]).values.T
    test_traffic_array = pd.DataFrame([test_time_flow_A_2['traffic_flow'],test_time_flow_A_3['traffic_flow'],test_time_flow_B_1['traffic_flow'],
                                       test_time_flow_B_3['traffic_flow'],test_time_flow_C_1['traffic_flow'],test_time_flow_C_3['traffic_flow']]).values.T

    # 对车流量进行归一化操作
    min_max_scalar = preprocessing.MinMaxScaler()
    total_traffic_array = np.concatenate([train_traffic_array, test_traffic_array], axis=0)
    normalized_traffic_array = min_max_scalar.fit_transform(total_traffic_array)
    normalized_train_traffic_array = normalized_traffic_array[:6552]
    normalized_test_traffic_array = normalized_traffic_array[6552:]

    return train_time_array, normalized_train_traffic_array, test_time_array, normalized_test_traffic_array




# 数据标准化
def data_normalization(data, a): # a is the coefficient of normalization method
    std_value = []               # type(a) = float
    mean_value = []
    for i in range(np.shape(data)[1]):
        mean_value.append(np.mean(data[:,i]))
        std_value.append(np.std(data[:,i]))
        data[:,i] =  0.5*(np.tanh(a*(data[:,i] - np.mean(data[:,i]))/np.std(data[:,i])) + 1)
    return data, mean_value, std_value, a


# 数据反标准化,转化为真实值
# a is the coefficient of normalization method
# type(a) = float
# 必须要是二维数组!!!
def data_denormalization(result, mean_value, std_value, a):        # shape(result) = (84,6), type(result) = np.array
    for i in range(np.shape(result)[1]):
        result[:,i] = (1/(2*a) * np.log(result[:,i]/( 1 - result[:,i]))) * std_value[i] + mean_value[i]
    return result


# 线性归一化
def liner_normalization(data):
    min_max_scalar = preprocessing.MinMaxScaler()
    normalized_data = min_max_scalar.fit_transform(data)
    return normalized_data, min_max_scalar.data_max_, min_max_scalar.data_min_


# 线性去归一化
# data只有一列数据的时候！
def liner_denormalization(data, data_max, data_min):
    result = np.zeros_like(data)
    for i in range(np.shape(data)[1]):
        result[:,i] = data[:,i]*(data_max - data_min) + data_min
    return result



# 将训练集和测试集在同一个标准下进行归一化, 返回标准化之后的训练集和测试集
def get_normalized_data(a):
    train_data = load_train_time()          # shape = (6552, 6)
    test_data = load_test_time()            # shape = (84, 6)
    All_data = np.concatenate([train_data, test_data], axis=0)  #shape = (6552+84, 6)
    normalized_data, mean_value, std_value, a = data_normalization(All_data, a)
    normalized_train_data = normalized_data[:6552]              # shape = (6552, 6)
    normalized_test_data = normalized_data[6552:]               # shape = (84, 6)

    return normalized_train_data, normalized_test_data, mean_value, std_value, a


# 线性归一化
def get_liner_normalizer_data():
    train_data = load_train_time()         # shape = (6552, 6)
    test_data = load_test_time()           # shape = (84, 6)
    All_data = np.concatenate([train_data, test_data], axis=0)  #shape = (6552+84, 6)
    normalized_data, data_max, data_min = liner_normalization(All_data)
    normalized_train_data = normalized_data[:6552]              # shape = (6552, 6)
    normalized_test_data = normalized_data[6552:]               # shape = (84, 6)
    return normalized_train_data, normalized_test_data, data_max, data_min




# 生成用于提交的DataFrame
def generate_submitted_result(result):   # shape(result) = (84,6), type(result) = np.array

    #读取提交样本中的time_window
    time_window = pd.read_csv('/home/johnson/PycharmProjects/KDD_CUP/average_time_results/gxy.csv')['time_window'].values
    intersection_id = ['A']*168 + ['B']*168 + ['C']*168
    tollgate_id = [2]*84 + [3]*84 + ([1]*84 + [3]*84) * 2
    avg_travel_time = np.ravel(result, 1)                           #将result按列展开
    result_DataFrame = pd.DataFrame({
                                     'intersection_id':intersection_id,
                                     'tollgate_id':tollgate_id,
                                     'time_window':time_window,
                                     'avg_travel_time':avg_travel_time
                                     })
    return result_DataFrame




# 计算MAPE, 与题中的误差度量相对应
def cal_MAPE(target, predict):    # shape(target) = shape(predict) = (6,6) = (num_steps, output_size),  type = np.array
    target = np.reshape(np.array(target), [-1,])
    predict = np.reshape(np.array(predict), [-1,])
    MAPE = (np.abs(target - predict)/target).mean()
    return MAPE


# 计算MAE, 避免的分母中0的情况. 用MAE近似代替MAPE
def cal_MAE(target, predict):
    target = np.reshape(np.array(target), [-1,])
    predict = np.reshape(np.array(predict), [-1,])
    MAE = np.abs(target - predict).mean()
    return MAE


## --------------------------------------new data_rader不断的向神经网络喂数据------------------------------------------##
########################################################################################################################################

# 每条路径单独预测, 输入的参数a为整型, 代表第几条路径
def load_train_test_data(a):
    # train data
    nor_train_traffic_data = load_train_traffic()                               # shape = (6552, 6)
    nor_train_time_data, _, data_max, data_min = get_liner_normalizer_data()    # shape = (6556, 6)
    train_input_data = np.stack([nor_train_time_data[:,a], nor_train_traffic_data[:,a]], axis=1) # shape = (6556, 2)
    #train_output_data = load_train_time()[:,a]                                  # shape = (6556, 1), 没有归一化！
    train_output_data = nor_train_time_data[:,a]

    # test data
    nor_test_traffic_data = load_test_traffic()                                                 # shape = (84, 6)
    _, nor_test_time_data, data_max, data_min = get_liner_normalizer_data()                     # shape = (84,6)
    test_input_data = np.stack([nor_test_time_data[:,a], nor_test_traffic_data[:,a]], axis=1)   # shape = (84,2)

    return train_input_data,train_output_data, test_input_data



# 生成训练集的特征和标签, 输入input_data为指定路径的平均时间和车流量信息, output_data为指定路径的平均时间信息
# 返回的特征归一化, 标签未经过归一化
def generate_train_feature_label(input_data, output_data, holiday_data = None, weather_data = None):
    num_steps = 6
    data_length = np.shape(input_data)[0]                     # 数据集的长度
    input_size = 3
    other_feature_size = 0                                    # 其他特征的长度
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
        data_feature[i] = np.concatenate([feature1, feature2, feature6], axis=0)                                           # 总特征
        data_label[i] = np.reshape(output_data[data_length_index[i + num_steps: i + 2*num_steps]], [-1, ])

    # shape(data_feature) = (6540, 18), shape(data_label) = (6540, 6)
    return data_feature, data_label


# 生成最终提交测试集的特征, 输入input为指定路段测试的时间数据, holiday_data和weather_data均为测试集的节假日信息和温度信息
def generate_test_feature(test_input_data, holiday_data = None, weather_data = None):                              # shape(weather_data) = (84,4)
    train_load_features, test_load_features = get_load_features()                                                  # shape(test_load_features) = (84, 1)
    feature1 = np.reshape(test_input_data[:,0], [14, -1])                                                          # 前两小时的平均时间特征
    feature2 = np.reshape(test_input_data[:,1], [14, -1])                                                          #  前两小时的车流了特征
    feature3 = holiday_data                                                                                        # 节假日特征 shape = (14,2)
    feature4 = np.array(weather_data[::6, 2:])                                                                     # 加入温度和湿度特征
    feature6 = np.reshape(test_load_features[:,0], [14, -1])                                                       # shape = (14, 6)
    test_feature = np.concatenate([feature1, feature2, feature6], axis=1)
    return test_feature                                                                                            # shape = (14,18)



# 得到白天的索引
def filter_night_index():
    day_index_array = np.arange(18,54)  # 一天里属于白天的索引
    day_index = list(day_index_array)
    total_days = 90                     # 从7月19号到10月17号总天数
    for i in range(total_days):
        day_index_array += 72
        day_index.extend(list(day_index_array))
    return day_index
# 去除晚上和节假日的索引
def filter_night_holiday_index():
    holiday_index = []
    holiday_start = 74
    holidat_stop = 81
    for i in range(holiday_start, holidat_stop):
        holiday_index.extend(list(range(i*72,(i+1)*72)))

    day_index_array = np.arange(18,54)  # 一天里属于白天的索引
    day_index = list(day_index_array)
    total_days = 90                     # 从7月19号到10月17号总天数
    for i in range(total_days):
        day_index_array += 72
        day_index.extend(list(day_index_array))

    No_holiday_night_index = list(set(day_index) - set(holiday_index))
    return No_holiday_night_index
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





# 生成每条路径的特征和标签, 并shuffleh后生成训练集和测试集
def load_route_feature_label(a):
    day_index = filter_holiday_index()
    train_holiday_data = load_train_holidays()                                              # shape = (6552, 2)
    test_holiday_data = load_test_holidays()                                                # shape = (84, 2)
    normalized_train_weather_data, normalized_test_weather_data = load_weather_data()
    train_input_data,train_output_data, test_input_data = load_train_test_data(a)           # shape(train_input_data) = (6552, 2)
    # 训练数据上的特征和标签                                                                   # shape(train_output_data) = (6552, 1)
    data_feature, data_label = generate_train_feature_label(train_input_data, train_output_data, holiday_data=train_holiday_data, weather_data=normalized_train_weather_data)
    # 得到白天的特征和标签
    # data_feature = data_feature[day_index]
    # data_label = data_label[day_index]
    # 生成本地训练集和测试集
    train_feature, test_feature, train_label, test_label = train_test_split(data_feature, data_label, test_size=0, random_state=10)
    # shape(train_feature) = [?, 18], shape(train_label) = [?, 6]
    # shape(test_feature) = [?, 18], shape(test_label) = [?, 6]
    return train_feature, test_feature, train_label, test_label


# 生成最终用于提交的特征(针对每条路径)
def load_route_test_feature(a):
    test_holiday_data = load_test_holidays()                                                # shape = (84, 2)
    normalized_train_weather_data, normalized_test_weather_data = load_weather_data()
    train_input_data,train_output_data, test_input_data = load_train_test_data(a)           # shape(train_input_data) = (6552, 2)
    test_feature = generate_test_feature(test_input_data=test_input_data, holiday_data=test_holiday_data, weather_data=normalized_test_weather_data)
    return test_feature                                                                     # shape = (14, 18)



##------------------------------stacked AutoEncoder 提取道路中的信息----------------------------------------------------------------------###

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

###-------------------------------------------------------------------------------------------------------------------------------------####


# 定义一个类,将训练样本拆分成一个一个的batch
class batch_reader(object):
    def __init__(self, train_feature, test_feature, train_label, test_label, batch_size, is_training):
        self.train_feature = train_feature
        self.test_feature = test_feature
        self.train_label = train_label
        self.test_label = test_label
        self.batch_size = batch_size
        self.is_training = is_training
        self.train_batch_start = 0
        self.test_batch_start = 0
        if self.is_training:
            self.num_batches = len(self.train_feature)/self.batch_size
        else:
            self.num_batches = len(self.test_feature)/self.batch_size

    # 得到下一个batch的数据
    def next_batch(self):
        if self.is_training:
            self.train_feature_batch = self.train_feature[self.train_batch_start: self.train_batch_start + self.batch_size]
            self.train_label_batch = self.train_label[self.train_batch_start: self.train_batch_start + self.batch_size]
            self.train_batch_start += self.batch_size
            return self.train_feature_batch, self.train_label_batch
        else:
            self.test_data_batch = self.test_feature[self.test_batch_start: self.test_batch_start + self.batch_size]
            self.test_label_batch = self.test_label[self.test_batch_start : self.test_batch_start + self.batch_size]
            self.test_batch_start += self.batch_size
            return self.test_data_batch, self.test_label_batch

    # 重置数据
    def reset(self):
        self.train_batch_start = 0
        self.test_batch_start = 0





###################################################################################################################################################










# 通过定义一个类, 实现不断的向神经网络喂数据
class data_reader(object):
    def __init__(self, input_data, output_data, batch_size, num_steps, input_size, output_size, split_rate, is_training):
        self.batch_size = batch_size
        self.num_steps = num_steps                          # 数据截断长度
        self.input_size = input_size
        self.output_size = output_size
        self.input_data = input_data                        # shape = (6552, ?)     ? 对应这特征的维数, 输入的特征经过归一化
        self.output_data = output_data                      # shape = (6552, 6)     输出的label值未经过归一化
        self.split_rate = split_rate
        self.is_training = is_training                      # 用于判断是否在训练
        self.train_batch_start = 0                          # 训练集的起始位置
        self.test_batch_start = 0                           # 测试集的起始位置
        self.get_train_test_set()
        self.next_batch()



    # 循环得到训练集和测试集
    def get_train_test_set(self):
        data_length = np.shape(self.input_data)[0]                # 数据集的长度
        test_length = int(data_length * self.split_rate)          # 测试集的长度
        train_length = data_length - test_length                  # 训练集的长度
        train_length_index = np.arange(train_length)
        test_length_index = np.arange(test_length)
        self.train_data = np.zeros((train_length - 2*self.num_steps, self.num_steps, self.input_size))    # 初始化训练集（输入）
        self.train_label = np.zeros((train_length - 2*self.num_steps, self.num_steps, self.output_size))  # 初始化训练集（预测值）
        self.test_data = np.zeros((test_length - 2*self.num_steps, self.num_steps, self.input_size))      # 初始化测试集（输入）
        self.test_label = np.zeros((test_length - 2*self.num_steps, self.num_steps, self.output_size))    # 初始化测试集（预测值）

        if self.is_training:
            self.num_batches = int(np.floor((train_length - 2*self.num_steps)/self.batch_size))
        else:
            self.num_batches = int(np.floor((test_length - 2*self.num_steps)/self.batch_size))

        for i in range(train_length - 2*self.num_steps):
            self.train_data[i] = self.input_data[train_length_index[i:i + self.num_steps]]
            self.train_label[i] = self.output_data[train_length_index[i + self.num_steps: i + 2*self.num_steps]][np.newaxis].T  # 增加一个维度

        for i in range(test_length - 2*self.num_steps):
            self.test_data[i] = self.input_data[train_length + test_length_index[i:i + self.num_steps]]
            self.test_label[i] = self.output_data[train_length + test_length_index[i + self.num_steps: i + 2*self.num_steps]][np.newaxis].T # 增加一个维度

        return self.train_data, self.train_label, self.test_data, self.test_label



    # 返回数据和标签(分别作为输入和输出)
    def next_batch(self):
        if self.is_training:
            self.train_data_batch = self.train_data[self.train_batch_start: self.train_batch_start + self.batch_size]
            self.train_label_batch = self.train_label[self.train_batch_start: self.train_batch_start + self.batch_size]
            self.train_batch_start += self.batch_size
            return self.train_data_batch, self.train_label_batch
        else:
            self.test_data_batch = self.test_data[self.test_batch_start: self.test_batch_start + self.batch_size]
            self.test_label_batch = self.test_label[self.test_batch_start : self.test_batch_start + self.batch_size]
            self.test_batch_start += self.batch_size
            return self.test_data_batch, self.test_label_batch

    # 重置数据
    def reset(self):
        self.train_batch_start = 0
        self.test_batch_start = 0



# # 测试数据归一化与去归一化
# train_data = load_train_time()
#
# nor_train_data, data_max, data_min = liner_normalization(train_data)
#
#
# denor_train_data = liner_denormalization(nor_train_data, data_max, data_min)
#
# print denor_train_data - train_data
#
# plt.plot(nor_train_data[:,0])
#
#
# plt.show()

# a,b,c,d = load_route_feature_label(0)
#
# cs = batch_reader(a,b,c,d,128, is_training=True)
#
# e,f = cs.next_batch()
#
# print e.shape
# print f.shape
# print e[0]
# print f[0]




