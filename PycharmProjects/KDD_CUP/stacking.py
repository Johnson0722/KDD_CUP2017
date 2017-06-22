import pandas as pd
import numpy as np
from utils import *


result1 = pd.read_csv('average_time_results/by_route_add_weather_model_fusion_all_data.csv') # MAPE = 0.1965

result2 = pd.read_csv('average_time_results/by_route_add_weather_load_model_fusion_filter_holiday.csv') # 0.1955

result3 = pd.read_csv('average_time_results/by_route_filter_holiday_add_load_lstm_data_fusion.csv') # 0.1996

k1 = 0

k2 = 0.8

k3 = 0.2

predict_value = result1['avg_travel_time'].values * k1 + result2['avg_travel_time'].values * k2 + result3['avg_travel_time'].values * k3

result_DataFrame = generate_submitted_result(predict_value)
result_DataFrame.to_csv('average_time_results/stacking_linear_LSTM_filter_holiday_add_load_5_8.csv', index=False, columns=['intersection_id','tollgate_id','time_window','avg_travel_time'])

