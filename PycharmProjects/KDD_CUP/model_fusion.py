from linear_model import get_all_time_results
from filter_night_hours import get_day_time_results
from utils import *
import numpy as np
import pandas as pd


#--------------linear regression fusion---------------##

best_A_2, best_A_3, day_time_predict_B_1, best_B_3, day_time_predict_C_1, day_time_predict_C_3 = get_day_time_results()
best_B_1 = pd.read_csv('average_time_results/result_by_route/best_B_1.csv', index_col=0, header=None)
best_C_1 = pd.read_csv('average_time_results/result_by_route/best_C_1.csv', index_col=0, header=None)
best_C_3 = pd.read_csv('average_time_results/result_by_route/best_C_3.csv', index_col=0, header=None)

best_B_1 = best_B_1.values.flatten()
best_C_1 = best_C_1.values.flatten()
best_C_3 = best_C_3.values.flatten()

predict_value = np.stack([best_A_2, best_A_3, best_B_1, best_B_3, best_C_1, best_C_3], axis=1)        # shape = (84, 6)

result_DataFrame = generate_submitted_result(predict_value)
result_DataFrame.to_csv('average_time_results/best_result.csv', index=False, columns=['intersection_id','tollgate_id','time_window','avg_travel_time'])



# #-----------------LSTM fusion-------------------##
# All_time_A_2 = pd.read_csv('average_time_results/result_by_route/All_day_A_2.csv', index_col=0, header=None)
# All_time_A_3 = pd.read_csv('average_time_results/result_by_route/All_day_A_3.csv', index_col=0, header=None)
# All_time_B_1 = pd.read_csv('average_time_results/result_by_route/All_day_B_1.csv', index_col=0, header=None)
# All_time_B_3 = pd.read_csv('average_time_results/result_by_route/All_day_B_3.csv', index_col=0, header=None)
# All_time_C_1 = pd.read_csv('average_time_results/result_by_route/All_day_C_1.csv', index_col=0, header=None)
# All_time_C_3 = pd.read_csv('average_time_results/result_by_route/All_day_C_3.csv', index_col=0, header=None)
#
# day_time_A_2 = pd.read_csv('average_time_results/result_by_route/day_time_A_2.csv', index_col=0, header=None)
# day_time_A_3 = pd.read_csv('average_time_results/result_by_route/day_time_A_3.csv', index_col=0, header=None)
# day_time_B_1 = pd.read_csv('average_time_results/result_by_route/day_time_B_1.csv', index_col=0, header=None)
# day_time_B_3 = pd.read_csv('average_time_results/result_by_route/day_time_B_3.csv', index_col=0, header=None)
# day_time_C_1 = pd.read_csv('average_time_results/result_by_route/day_time_C_1.csv', index_col=0, header=None)
# day_time_C_3 = pd.read_csv('average_time_results/result_by_route/day_time_C_3.csv', index_col=0, header=None)
#
#
# # predict_value = np.concatenate([All_time_A_2.values, All_time_A_3.values, All_time_B_1.values, All_time_B_3.values, All_time_C_1.values, All_time_C_3.values], axis=1)    # shape = (84, 6)
# predict_value = np.concatenate([day_time_A_2.values, day_time_A_3.values, All_time_B_1.values, day_time_B_3.values, All_time_C_1.values, All_time_C_3.values], axis=1)    # shape = (84, 6)
#
#
# result_DataFrame = generate_submitted_result(predict_value)
# result_DataFrame.to_csv('average_time_results/by_route_lstm_data_fusion.csv', index=False, columns=['intersection_id','tollgate_id','time_window','avg_travel_time'])



# #-----------------filter holiday LSTM fusion-------------------##
# A_2 = pd.read_csv('average_time_results/result_by_route/filter_holiday_add_load_A_2.csv', index_col=0, header=None)
# A_3 = pd.read_csv('average_time_results/result_by_route/filter_holiday_add_load_A_3.csv', index_col=0, header=None)
# B_1 = pd.read_csv('average_time_results/result_by_route/filter_holiday_add_load_B_1.csv', index_col=0, header=None)
# B_3 = pd.read_csv('average_time_results/result_by_route/filter_holiday_add_load_B_3.csv', index_col=0, header=None)
# C_1 = pd.read_csv('average_time_results/result_by_route/filter_holiday_add_load_C_1.csv', index_col=0, header=None)
# C_3 = pd.read_csv('average_time_results/result_by_route/filter_holiday_add_load_C_3.csv', index_col=0, header=None)
#
#
#
# # predict_value = np.concatenate([All_time_A_2.values, All_time_A_3.values, All_time_B_1.values, All_time_B_3.values, All_time_C_1.values, All_time_C_3.values], axis=1)    # shape = (84, 6)
# predict_value = np.concatenate([A_2.values, A_3.values, B_1.values, B_3.values, C_1.values, C_3.values], axis=1)    # shape = (84, 6)
#
#
# result_DataFrame = generate_submitted_result(predict_value)
# result_DataFrame.to_csv('average_time_results/by_route_filter_holiday_add_load_lstm_data_fusion.csv', index=False, columns=['intersection_id','tollgate_id','time_window','avg_travel_time'])
