#coding:utf-8
import pandas as pd
from dateutil.parser import parse
import numpy as np

#  读取天气数据,使用date字段作为索引
weather_data = pd.read_csv("/home/johnson/PycharmProjects/KDD_CUP/dataSets/dataSets/testing_phase1/weather (table 7)_test1.csv", index_col='date')

# print weather_data.info()       ## 没有缺失值

# 生成新的日期索引, parse模块将字符串解析为一个日期
new_index = []
for i in range(len(weather_data.index)):
    date_time = parse(str(weather_data.index[i]) + ' ' + str(weather_data['hour'][i]) + ':00' + ':00')
    new_index.append(date_time)

# 用新索引替换
weather_data.index = new_index

# print weather_data


# 生成匹配的日期范围
date = pd.date_range(start = '10/18/2016 00:00:00', end = '10/24/2016 21:00:00', freq = '20min')

# 重新索引,向前填充
weather_data =  weather_data.reindex(index = date, method = 'ffill')

# 去除无用字段
weather_data.drop(['hour', 'pressure', 'sea_pressure', 'precipitation'], axis = 1, inplace=True)

# 处理异常数据, 将其置为Nan
weather_data[weather_data['wind_direction'] > 360] = np.nan

# 向前填充Nan值
weather_data.fillna(method='ffill', inplace= True)


# 写入文件
weather_data.to_csv('preprocessed_data/test_data/test_weather_date.csv')
