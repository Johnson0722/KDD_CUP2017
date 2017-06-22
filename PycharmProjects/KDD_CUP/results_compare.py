import pandas as pd
import numpy as np


result8 = pd.read_csv('average_time_results/by_route_add_weather_model_fusion_all_data.csv')    # 19.65

result9 = pd.read_csv('average_time_results/by_route_lstm_data_fusion.csv') # 20.40

result10 = pd.read_csv('average_time_results/stacking_linear_LSTM.csv')  #19.81

result11 = pd.read_csv('average_time_results/by_route_add_weather_model_fusion_filter_holiday.csv')

result12 = pd.read_csv('average_time_results/by_route_add_weather_load_model_fusion_filter_holiday.csv') # 19.55

result13 = pd.read_csv('average_time_results/by_route_filter_holiday_add_load_lstm_data_fusion.csv') # 19.96

result14 = pd.read_csv('average_time_results/stacking_linear_LSTM_filter_holiday_add_load.csv') # 19.52

result15 = pd.read_csv('average_time_results/best_result.csv')  # 0.2018


print np.mean(np.abs(result12['avg_travel_time'] - result8['avg_travel_time']))

print np.mean(np.abs(result12['avg_travel_time'] - result9['avg_travel_time']))

print np.mean(np.abs(result12['avg_travel_time'] - result10['avg_travel_time']))

print np.mean(np.abs(result12['avg_travel_time'] - result11['avg_travel_time']))

print '---------------------------'

print np.mean(np.abs(result13['avg_travel_time'] - result8['avg_travel_time']))

print np.mean(np.abs(result13['avg_travel_time'] - result9['avg_travel_time']))

print np.mean(np.abs(result13['avg_travel_time'] - result10['avg_travel_time']))

print np.mean(np.abs(result13['avg_travel_time'] - result11['avg_travel_time']))

print np.mean(np.abs(result13['avg_travel_time'] - result12['avg_travel_time']))

print '-----------------------'
print np.mean(np.abs(result14['avg_travel_time'] - result8['avg_travel_time']))

print np.mean(np.abs(result14['avg_travel_time'] - result9['avg_travel_time']))

print np.mean(np.abs(result14['avg_travel_time'] - result10['avg_travel_time']))

print np.mean(np.abs(result14['avg_travel_time'] - result11['avg_travel_time']))

print np.mean(np.abs(result14['avg_travel_time'] - result12['avg_travel_time']))

print np.mean(np.abs(result14['avg_travel_time'] - result13['avg_travel_time']))

print '-----------------------'

print np.mean(np.abs(result15['avg_travel_time'] - result8['avg_travel_time']))

print np.mean(np.abs(result15['avg_travel_time'] - result9['avg_travel_time']))

print np.mean(np.abs(result15['avg_travel_time'] - result10['avg_travel_time']))

print np.mean(np.abs(result15['avg_travel_time'] - result11['avg_travel_time']))

print np.mean(np.abs(result15['avg_travel_time'] - result12['avg_travel_time']))

print np.mean(np.abs(result15['avg_travel_time'] - result13['avg_travel_time']))

print np.mean(np.abs(result15['avg_travel_time'] - result14['avg_travel_time']))






