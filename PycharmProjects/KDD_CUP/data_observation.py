import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


A_2 = pd.read_csv('/home/johnson/PycharmProjects/KDD_CUP/preprocessed_data/train_data/time_flow_A_2.csv',index_col=0)
A_3 = pd.read_csv('/home/johnson/PycharmProjects/KDD_CUP/preprocessed_data/train_data/time_flow_A_3.csv',index_col=0)
B_1 = pd.read_csv('/home/johnson/PycharmProjects/KDD_CUP/preprocessed_data/train_data/time_flow_B_1.csv',index_col=0)
B_3 = pd.read_csv('/home/johnson/PycharmProjects/KDD_CUP/preprocessed_data/train_data/time_flow_B_3.csv',index_col=0)
C_1 = pd.read_csv('/home/johnson/PycharmProjects/KDD_CUP/preprocessed_data/train_data/time_flow_C_1.csv',index_col=0)
C_3 = pd.read_csv('/home/johnson/PycharmProjects/KDD_CUP/preprocessed_data/train_data/time_flow_C_3.csv',index_col=0)

test_A_2 = pd.read_csv('/home/johnson/PycharmProjects/KDD_CUP/preprocessed_data/test_data/test_time_flow_A_2.csv',index_col=0)
test_A_3 = pd.read_csv('/home/johnson/PycharmProjects/KDD_CUP/preprocessed_data/test_data/test_time_flow_A_3.csv',index_col=0)
test_B_1 = pd.read_csv('/home/johnson/PycharmProjects/KDD_CUP/preprocessed_data/test_data/test_time_flow_B_1.csv',index_col=0)
test_B_3 = pd.read_csv('/home/johnson/PycharmProjects/KDD_CUP/preprocessed_data/test_data/test_time_flow_B_3.csv',index_col=0)
test_C_1 = pd.read_csv('/home/johnson/PycharmProjects/KDD_CUP/preprocessed_data/test_data/test_time_flow_C_1.csv',index_col=0)
test_C_3 = pd.read_csv('/home/johnson/PycharmProjects/KDD_CUP/preprocessed_data/test_data/test_time_flow_C_3.csv',index_col=0)


# travel_df = pd.DataFrame([A_2['travel_time'].T, A_3['travel_time'].T, B_1['travel_time'].T, B_3['travel_time'].T, C_1['travel_time'].T, C_3['travel_time'].T])
# travel_df = travel_df.T
# print travel_df.corr()

print A_2['travel_time'][0:72].describe()
print A_2['travel_time'][72:144].describe()
print A_2['travel_time'][72*2:72*3].describe()
print A_2['travel_time'][72*3:72*4].describe()
print A_2['travel_time'][72*4:72*5].describe()

print '----------------------National holiday-----------------------'
print A_2['travel_time'][72*74:72*75].describe()
print A_2['travel_time'][72*75:72*76].describe()
print A_2['travel_time'][72*76:72*77].describe()
print A_2['travel_time'][72*77:72*78].describe()
print A_2['travel_time'][72*78:72*79].describe()
print A_2['travel_time'][72*79:72*80].describe()
print A_2['travel_time'][72*80:72*81].describe()
print A_2['travel_time'][72*74:72*81].describe()
print '------------------------moon---------------------'
print A_2['travel_time'][72*68:72*69].describe()
print A_2['travel_time'][72*69:72*70].describe()
print A_2['travel_time'][72*70:72*71].describe()
print A_2['travel_time'][72*68:72*71].describe()
print '-----------------------holiday---------------------------'
print A_2['travel_time'][72*81:72*82].describe()
print A_2['travel_time'][72*90:72*82].describe()



# fig = plt.figure(1)
#
# ax1 = fig.add_subplot(3, 2, 1)
# ax2 = fig.add_subplot(3, 2, 2)
# ax3 = fig.add_subplot(3, 2, 3)
# ax4 = fig.add_subplot(3, 2, 4)
# ax5 = fig.add_subplot(3, 2, 5)
# ax6 = fig.add_subplot(3, 2, 6)
#
# ax1.plot(A_2['travel_time'], label='A_2')
# ax2.plot(A_3['travel_time'], label='A_3')
# ax3.plot(B_1['travel_time'], label='B_1')
# ax4.plot(B_3['travel_time'], label='B_3')
# ax5.plot(C_1['travel_time'], label='C_1')
# ax6.plot(C_3['travel_time'], label='C_3')
#
# ax1.legend(loc = 'best')
# ax2.legend(loc = 'best')
# ax3.legend(loc = 'best')
# ax4.legend(loc = 'best')
# ax5.legend(loc = 'best')
# ax6.legend(loc = 'best')


# fig = plt.figure(2)
#
#
# ax1 = fig.add_subplot(3, 2, 1)
# ax2 = fig.add_subplot(3, 2, 2)
# ax3 = fig.add_subplot(3, 2, 3)
# ax4 = fig.add_subplot(3, 2, 4)
# ax5 = fig.add_subplot(3, 2, 5)
# ax6 = fig.add_subplot(3, 2, 6)
#
# ax1.plot(A_2['traffic_flow'], label='A_2')
# ax2.plot(A_3['traffic_flow'], label='A_3')
# ax3.plot(B_1['traffic_flow'], label='B_1')
# ax4.plot(B_3['traffic_flow'], label='B_3')
# ax5.plot(C_1['traffic_flow'], label='C_1')
# ax6.plot(C_3['traffic_flow'], label='C_3')
#
#
# ax1.legend(loc = 'best')
# ax2.legend(loc = 'best')
# ax3.legend(loc = 'best')
# ax4.legend(loc = 'best')
# ax5.legend(loc = 'best')
# ax6.legend(loc = 'best')

# fig = plt.figure(3)
#
# ax1 = fig.add_subplot(3, 2, 1)
# ax2 = fig.add_subplot(3, 2, 2)
# ax3 = fig.add_subplot(3, 2, 3)
# ax4 = fig.add_subplot(3, 2, 4)
# ax5 = fig.add_subplot(3, 2, 5)
# ax6 = fig.add_subplot(3, 2, 6)
#
# ax1.plot(test_A_2['travel_time'], label='A_2')
# ax2.plot(test_A_3['travel_time'], label='A_3')
# ax3.plot(test_B_1['travel_time'], label='B_1')
# ax4.plot(test_B_3['travel_time'], label='B_3')
# ax5.plot(test_C_1['travel_time'], label='C_1')
# ax6.plot(test_C_3['travel_time'], label='C_3')
#
#
# ax1.legend(loc = 'best')
# ax2.legend(loc = 'best')
# ax3.legend(loc = 'best')
# ax4.legend(loc = 'best')
# ax5.legend(loc = 'best')
# ax6.legend(loc = 'best')
#
plt.show()
