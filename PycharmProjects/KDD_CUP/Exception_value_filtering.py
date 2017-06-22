# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.parser import parse
from pylab import *


trajectory_data = pd.read_csv('/home/johnson/PycharmProjects/KDD_CUP/dataSets/dataSets/training/trajectories(table 5)_training.csv',index_col = False)

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

print data_A_2['travel_time'].describe()
Q1 = data_A_2['travel_time'].quantile(0.25) # 下四分为数
Q3 = data_A_2['travel_time'].quantile(0.75) # 上四分为数
diff = Q3 - Q1
print Q3 + 1.5*diff
print Q1 - 1.5*diff

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


# # 画箱线图
# boxplot([data_A_2['travel_time'].values, data_A_3['travel_time'].values, data_B_1['travel_time'].values, data_B_3['travel_time'].values
#         ,data_C_1['travel_time'].values, data_C_3['travel_time'].values], labels=('A_2', 'A_3', 'B_1', 'B_3', 'C_1', 'C_3'))
#
# plt.show()


# # 异常值大于600的全部剔除
# data_A_2 = data_A_2[data_A_2['travel_time'] < 600]
# data_A_3 = data_A_3[data_A_3['travel_time'] < 600]
# data_B_1 = data_B_1[data_B_1['travel_time'] < 600]
# data_B_3 = data_B_3[data_B_3['travel_time'] < 600]
# data_C_1 = data_C_1[data_C_1['travel_time'] < 600]
# data_C_3 = data_C_3[data_C_3['travel_time'] < 600]
#
fig = plt.figure()

ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4)
ax5 = fig.add_subplot(3, 2, 5)
ax6 = fig.add_subplot(3, 2, 6)

ax1.plot(data_A_2['travel_time'], label='A_2')
ax2.plot(data_A_3['travel_time'], label='A_3')
ax3.plot(data_B_1['travel_time'], label='B_1')
ax4.plot(data_B_3['travel_time'], label='B_3')
ax5.plot(data_C_1['travel_time'], label='C_1')
ax6.plot(data_C_3['travel_time'], label='C_3')


ax1.legend(loc = 'best')
ax2.legend(loc = 'best')
ax3.legend(loc = 'best')
ax4.legend(loc = 'best')
ax5.legend(loc = 'best')
ax6.legend(loc = 'best')

plt.show()
