#coding: utf-8
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse
import matplotlib.pyplot as plt

trajectory_data = pd.read_csv('/home/johnson/PycharmProjects/KDD_CUP/dataSets/dataSets/training/trajectories(table 5)_training.csv',index_col = False)
# 打印trajectory_date的整体信息,是否有缺失值


# 筛选出各条路径的索引的索引, 生成关于每条路径的DateFrame

a1 = [trajectory_data['intersection_id'] == 'A'][0].values
b1 = [trajectory_data['tollgate_id'] == 3][0].values
index_A_3 = []
[index_A_3.append(x and y) for x,y in zip(a1,b1)]
data_A_3 = trajectory_data[index_A_3]


a2 = [trajectory_data['intersection_id'] == 'A'][0].values
b2 = [trajectory_data['tollgate_id'] == 2][0].values
index_A_2 = []
[index_A_2.append(x and y) for x,y in zip(a2,b2)]
data_A_2 = trajectory_data[index_A_2]



a3 = [trajectory_data['intersection_id'] == 'B'][0].values
b3 = [trajectory_data['tollgate_id'] == 1][0].values
index_B_1 = []
[index_B_1.append(x and y) for x,y in zip(a3,b3)]
data_B_1 = trajectory_data[index_B_1]


a4 = [trajectory_data['intersection_id'] == 'B'][0].values
b4 = [trajectory_data['tollgate_id'] == 3][0].values
index_B_3 = []
[index_B_3.append(x and y) for x,y in zip(a4,b4)]
data_B_3 = trajectory_data[index_B_3]


a5 = [trajectory_data['intersection_id'] == 'C'][0].values
b5 = [trajectory_data['tollgate_id'] == 1][0].values
index_C_1 = []
[index_C_1.append(x and y) for x,y in zip(a5,b5)]
data_C_1 = trajectory_data[index_C_1]


a6 = [trajectory_data['intersection_id'] == 'C'][0].values
b6 = [trajectory_data['tollgate_id'] == 3][0].values
index_C_3 = []
[index_C_3.append(x and y) for x,y in zip(a6,b6)]
data_C_3 = trajectory_data[index_C_3]


# 利用箱线图去除异常值, 大于Q3 + 1.5diff 和 小于Q1 - 1.5diff的被标记为异常值
def Exception_value_filtering(data):
    Q1 = data['travel_time'].quantile(0.25) # 下四分为数
    Q3 = data['travel_time'].quantile(0.75) # 上四分为数
    diff = Q3 - Q1
    data = data[data['travel_time'] < Q3 + 1.5*diff]
    data = data[data['travel_time'] > Q1 - 1.5*diff]
    return data

data_A_2 = Exception_value_filtering(data_A_2)
data_A_3 = Exception_value_filtering(data_A_3)
data_B_1 = Exception_value_filtering(data_B_1)
data_B_3 = Exception_value_filtering(data_B_3)
data_C_1 = Exception_value_filtering(data_C_1)
data_C_3 = Exception_value_filtering(data_C_3)


# 对数据进行聚合处理(时间和交通流量)
def preprocess_groupby(DateFrame, path):
    # drop irrelevent field
    DateFrame = DateFrame.drop('travel_seq',axis = 1)

    # transform start time into time window(20 mins)
    k_time_window = []                             # the k-th time interval(time window = 20min)
    [k_time_window.append(np.floor((parse(x) - datetime(2016,7,19,0,0,0)).total_seconds()/1200.0)) for x in DateFrame['starting_time'].values]
    DateFrame['starting_time'] = k_time_window     # change starting time to time window

    # get groupby average time
    average_time = DateFrame['travel_time'].groupby(DateFrame['starting_time']).mean() # type(average_time) = <class 'pandas.core.series.Series'>
    # get groupby traffic flow
    traffic_flow = DateFrame['vehicle_id'].groupby(DateFrame['starting_time']).count() # type(Traffic_flow) = <class 'pandas.core.series.Series'>

    # get zeros series
    zeros_series = pd.Series(np.zeros(6552), index=np.arange(6552))

    # fill index not exist
    average_time = average_time + zeros_series
    traffic_flow = traffic_flow + zeros_series
    average_time.interpolate(inplace=True)          # 对于在该窗口下没有车经过的情况,平均时间数据采用插值处理
    traffic_flow.interpolate(inplace=True)          # 对于在该窗口下没有车经过的情况,车流量数据采用插值处理
    average_time.fillna(method = 'bfill', inplace=True)
    traffic_flow.fillna(method = 'bfill', inplace=True)
    grouped_dataframe = pd.DataFrame({"travel_time": average_time,
                                      "traffic_flow": traffic_flow})
    # write to the files
    grouped_dataframe.to_csv(path)



preprocess_groupby(data_A_2,'preprocessed_data/train_data/time_flow_A_2.csv')   # len = 6552
preprocess_groupby(data_A_3,'preprocessed_data/train_data/time_flow_A_3.csv')   # len = 6552
preprocess_groupby(data_B_1,'preprocessed_data/train_data/time_flow_B_1.csv')   # len = 6552
preprocess_groupby(data_B_3,'preprocessed_data/train_data/time_flow_B_3.csv')   # len = 6552
preprocess_groupby(data_C_1,'preprocessed_data/train_data/time_flow_C_1.csv')   # len = 6552
preprocess_groupby(data_C_3,'preprocessed_data/train_data/time_flow_C_3.csv')   # len = 6552
