import psycopg2
import pandas as pd
import src.data as data
import csv
import re
import os
import glob
from sklearn.model_selection import train_test_split


# Merge Point Machine data
def merge_point_machine_data_from_db():
    point_machine_df = data.fetch_point_machine_data_from_db()
    point_machine_movement_data_df = data.fetch_point_machine_movement_data_from_db()
    merge_point_machine_data_df = pd.merge(point_machine_df, point_machine_movement_data_df, on='timestamp')
    return merge_point_machine_data_df


# Merge LC Data
def merge_lc_data_from_db():
    lc_df = data.fetch_lc_data_from_db()
    lc_movement_data_df = data.fetch_lc_movement_data_from_db()
    merge_lc_data_df = pd.merge(lc_df, lc_movement_data_df, on='generated_timestamp')
    return merge_lc_data_df

    # Load rbc  data


def merge_rbc_data_from_db():
    rbc_files = [f for f in os.listdir("./") if f.endswith(".csv") and 'rbc_log' in f]
    rbc_data_df = None
    rbc_data = None
    for file in rbc_files:
        rbc_data.append(pd.read_csv(file))
    rbc_data_df = pd.read_csv(rbc_data)
    rbc_train_speed_data = data.fetch_rbc_train_speed_data_from_db()
    rbc_data_df = pd.merge(rbc_data_df, rbc_train_speed_data, on='obuid')
    return rbc_data_df

    # Load rbc  data
# def merge_rbc_data_from_db():
#    rbc_data_df = data.fetch_rbc_train_speed_data_from_db()
#    lc_movement_data_df = fetch_lc_movement_data_from_db()
#    merge_rbc_data__df = pd.merge(rbc_data_df, lc_movement_data_df, on='rbctimestamp')
#    return merge_rbc_data__df

# def get_weather_data():
#     conn = psycopg2.connect("host=<hostname> dbname=<dbname> user=<username> password=<password>")
#     log_lc_details = pd.read_sql("SELECT * FROM log_lc_details", conn)
#     weather = pd.read_sql("SELECT * FROM weather", conn)
#     data = pd.merge(log_lc_details, weather, on='id', how='inner')
#     X = data[['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob', 'precipcover', 'preciptype', 'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir', 'pressure', 'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex', 'severerisk']]
#     y = data['lc_status']
#     conn.close()
#     return X, y
#
# def split_data(data):
#     X, y = data
#     return train_test_split(X, y, test_size=0.2, random_state=0)
#
#
#
# import psycopg2
# import pandas as pd
# import numpy as np
#
# # Connect to the database
# conn = psycopg2.connect("dbname=database_name user=user_name password=password host=host_name")
#
# # Execute the query
# df = pd.read_sql_query("SELECT * FROM weather", conn)
#
# # Close the connection
# conn.close()
#
# # Calculate mean temperature
# mean_temp = df["temp"].mean()
#
# # Calculate standard deviation of temperature
# std_dev_temp = df["temp"].std()
#
# # Calculate median temperature
# median_temp = df["temp"].median()
#
# # Calculate the mode of the temperature
# mode_temp = df["temp"].mode().get(0)
#
# # Calculate the probability of precipitation
# prob_precip = len(df[df["precip"] > 0]) / len(df)
#
# # Print the results
# print("Mean temperature: ", mean_temp)
# print("Standard deviation of temperature: ", std_dev_temp)
# print("Median temperature: ", median_temp)
# print("Mode of temperature: ", mode_temp)
# print("Probability of precipitation: ", prob_precip)
#
#
#
#
# import numpy as np
# import pandas as pd
# from datetime import datetime
#
# # Load the weather data into a pandas DataFrame
# weather = pd.read_sql_query("SELECT * FROM weather", con)
#
# # Extract time-based features
# weather['hour'] = weather['datetime'].dt.hour
# weather['day_of_week'] = weather['datetime'].dt.dayofweek
# weather['month'] = weather['datetime'].dt.month
# weather['year'] = weather['datetime'].dt.year
#
# # Extract weather-based features
# weather['avg_temp'] = (weather['tempmax'] + weather['tempmin']) / 2
# weather['avg_humidity'] = weather['humidity'].mean()
# weather['avg_wind_speed'] = weather['windspeed'].mean()
# weather['avg_cloud_cover'] = weather['cloudcover'].mean()
# weather['precip_total'] = weather['precip'].sum()
# weather['snow_total'] = weather['snow'].sum()
# weather['temp_diff'] = weather['tempmax'] - weather['tempmin']
# weather['extreme_temp'] = np.where(weather['temp_diff'] > 10, 1, 0)
# weather['precip_3days_total'] = weather['precip'].rolling(3).sum()
# weather['snow_3days_total'] = weather['snow'].rolling(3).sum()
# weather['temp_rolling_mean'] = weather['temp'].rolling(3).mean()
# weather['temp_rolling_std'] = weather['temp'].rolling(3).std()
# weather['humidity_rolling_mean'] = weather['humidity'].rolling(3).mean()
# weather['humidity_rolling_std'] = weather['humidity'].rolling(3).std()
# weather['wind_speed_rolling_mean'] = weather['windspeed'].rolling(3).mean()
# weather['wind_speed_rolling_std'] = weather['windspeed'].rolling(3).std()
# weather['cloud_cover_rolling_mean'] = weather['cloudcover'].rolling(3).mean()
# weather['cloud_cover_rolling_std'] = weather['cloudcover'].rolling(3).std()
#
# # Calculate statistical parameters
# weather_corr = weather[['temp', 'humidity', 'windspeed', 'cloudcover']].corr()
# weather_cov = weather[['temp', 'humidity', 'windspeed', 'cloudcover']].cov()
#
# # Calculate probability parameters
# weather['rain_prob'] = weather[weather['preciptype'] == 'rain']['precipprob'].mean()
# weather['snow_prob'] = weather[weather['preciptype'] == 'snow']['precipprob'].mean()
# weather['severe_prob'] = weather[weather['severerisk'] > 0]['severerisk'].mean()
# weather['sunny_prob'] = weather[weather['cloudcover'] == 0]['cloudcover'].count() / weather['cloudcover'].count()
# weather['cloudy_prob'] = weather[(weather['cloudcover']
#
#
#
#
# import numpy as np
#
# # Thresholds for wind speed (in m/s)
# wind_speed_thresholds = {
#     'low': 0,
#     'moderate': 10,
#     'high': 20,
#     'very_high': 30,
#     'extreme': 50
# }
#
# # Thresholds for snow depth (in cm)
# snow_depth_thresholds = {
#     'low': 0,
#     'moderate': 10,
#     'high': 30,
#     'very_high': 50,
#     'extreme': 100
# }
#
# def categorize_wind_speed(wind_speed):
#     """
#     Categorize wind speed into low, moderate, high, very high, or extreme.
#     """
#     if wind_speed < wind_speed_thresholds['low']:
#         return 'low'
#     elif wind_speed >= wind_speed_thresholds['low'] and wind_speed < wind_speed_thresholds['moderate']:
#         return 'moderate'
#     elif wind_speed >= wind_speed_thresholds['moderate'] and wind_speed < wind_speed_thresholds['high']:
#         return 'high'
#     elif wind_speed >= wind_speed_thresholds['high'] and wind_speed < wind_speed_thresholds['very_high']:
#         return 'very_high'
#     elif wind_speed >= wind_speed_thresholds['very_high']:
#         return 'extreme'
#
# def categorize_snow_depth(snow_depth):
#     """
#     Categorize snow depth into low, moderate, high, very high, or extreme.
#     """
#     if snow_depth < snow_depth_thresholds['low']:
#         return 'low'
#     elif snow_depth >= snow_depth_thresholds['low'] and snow_depth < snow_depth_thresholds['moderate']:
#         return 'moderate'
#     elif snow_depth >= snow_depth_thresholds['moderate'] and snow_depth < snow_depth_thresholds['high']:
#         return 'high'
#     elif snow_depth >= snow_depth_thresholds['high'] and snow_depth < snow_depth_thresholds['very_high']:
#         return 'very_high'
#     elif snow_depth >= snow_depth_thresholds['very_high']:
#         return 'extreme'
#
# # Example usage:
# wind_speed = 25
# snow_depth = 70
#
# wind_speed_category = categorize_wind_speed(wind_speed)
# snow_depth_category = categorize_snow_depth(snow_depth)
#
# print(f"Wind speed: {wind_speed} m/s ({wind_speed_category})")
# print(f"Snow depth: {snow_depth} cm ({snow_depth_category})")
#
#
#
# import pandas as pd
#
# # Example categorical data
# data = {'precipType': ['rain', 'snow', 'rain', 'none']}
# df = pd.DataFrame(data)
#
# # One-hot encoding
# one_hot = pd.get_dummies(df['precipType'])
# df = df.drop('precipType', axis=1)
# df = df.join(one_hot)
#
# # Result
# print(df)
#
#
#
#
# import pandas as pd
#
# # Example precipitation data
# data = {'datetime': ['2021-01-01 00:00:00', '2021-01-01 01:00:00', '2021-01-01 02:00:00', '2021-01-01 03:00:00'],
#         'precip': [0.1, 0.2, 0.3, 0.4]}
# df = pd.DataFrame(data)
#
# # Convert datetime column to datetime type
# df['datetime'] = pd.to_datetime(df['datetime'])
#
# # Group by day and sum precipitation for each day
# df = df.groupby(df['datetime'].dt.date).sum()
#
# # Result
# print(df)
#
#
#
#
# import pandas as pd
#
# # load the data into a Pandas DataFrame
# df = pd.read_sql("select * from weather", con)
#
# # convert the datetime column to a DatetimeIndex
# df['datetime'] = pd.to_datetime(df['datetime'])
# df = df.set_index('datetime')
#
# # calculate the seasonal mean and standard deviation of the temperature, humidity, wind speed, and pressure
# seasonal_mean_temp = df['temp'].resample('M').mean()
# seasonal_std_temp = df['temp'].resample('M').std()
# seasonal_mean_humidity = df['humidity'].resample('M').mean()
# seasonal_std_humidity = df['humidity'].resample('M').std()
# seasonal_mean_windspeed = df['windspeed'].resample('M').mean()
# seasonal_std_windspeed = df['windspeed'].resample('M').std()
# seasonal_mean_pressure = df['pressure'].resample('M').mean()
# seasonal_std_pressure = df['pressure'].resample('M').std()
#
# # calculate the upper and lower bounds for each parameter based on the seasonal mean and standard deviation
# upper_bound_temp = seasonal_mean_temp + 3 * seasonal_std_temp
# lower_bound_temp = seasonal_mean_temp - 3 * seasonal_std_temp
# upper_bound_humidity = seasonal_mean_humidity + 3 * seasonal_std_humidity
# lower_bound_humidity = seasonal_mean_humidity - 3 * seasonal_std_humidity
# upper_bound_windspeed = seasonal_mean_windspeed + 3 * seasonal_std_windspeed
# lower_bound_windspeed = seasonal_mean_windspeed - 3 * seasonal_std_windspeed
# upper_bound_pressure = seasonal_mean_pressure + 3 * seasonal_std_pressure
# lower_bound_pressure = seasonal_mean_pressure - 3 * seasonal_std_pressure
#
# # plot the results to visualize the seasonality patterns
# import matplotlib.pyplot as plt
#
# plt.plot(seasonal_mean_temp, label='Seasonal Mean Temperature')
# plt.fill_between(seasonal_mean_temp.index, upper_bound_temp, lower_bound_temp, alpha=0.2, label='3-sigma Bounds')
# plt.legend()
# plt.show()
#
# plt.plot(seasonal_mean_humidity, label='Seasonal Mean Humidity')
# plt.fill_between(seasonal_mean_humidity.index, upper_bound_humidity, lower_bound_humidity, alpha=0.2, label='3-sigma Bounds')
# plt.legend()
# plt.show()
#
# plt.plot(seasonal_mean_windspeed, label='Seasonal Mean Wind Speed')
# plt.fill_between(seasonal_mean_windspeed.index, upper_bound_windspeed, lower_bound_windspeed, alpha=0.2, label='3-sigma Bounds')
# plt.legend()
# plt.show()
#
#
#
#
#
# import pandas as pd
# import numpy as np
#
# # Load data into a pandas DataFrame
# weather = pd.read_csv('weather_data.csv')
#
# # Feature Engineering
# weather['datetime'] = pd.to_datetime(weather['datetime'])
# weather['month'] = weather['datetime'].dt.month
# weather['day'] = weather['datetime'].dt.day
# weather['hour'] = weather['datetime'].dt.hour
#
# # Calculate the temperature range
# weather['temp_range'] = weather['tempmax'] - weather['tempmin']
#
# # Thresholds for temperature range
# temp_range_threshold = np.percentile(weather['temp_range'], [25, 75])
#
# # Create new columns for temperature range categories
# weather['temp_range_cold'] = (weather['temp_range'] <= temp_range_threshold[0]).astype(int)
# weather['temp_range_moderate'] = ((weather['temp_range'] > temp_range_threshold[0]) & (weather['temp_range'] <= temp_range_threshold[1])).astype(int)
# weather['temp_range_hot'] = (weather['temp_range'] > temp_range_threshold[1]).astype(int)
#
# # Thresholds for cloudcover
# cloudcover_threshold = np.percentile(weather['cloudcover'], [50])
#
# # Create a new column for cloudy or not
# weather['cloudy_prob'] = (weather['cloudcover'] >= cloudcover_threshold[0]).astype(int)
#
# import pandas as pd
# import numpy as np
#
#
# # Create a function to calculate the feature engineering
# def feature_engineering(weather):
#     # One hot encoding
#     conditions = weather['conditions'].str.get_dummies(sep=", ")
#     weather = weather.join(conditions)
#
#     # Calculate the threshold values
#     precip_threshold = weather['precip'].mean() + 2 * weather['precip'].std()
#     humidity_threshold = weather['humidity'].mean() + 2 * weather['humidity'].std()
#     windspeed_threshold = weather['windspeed'].mean() + 2 * weather['windspeed'].std()
#
#     # Apply one-hot-encoding to create new features
#     weather['precip_excessive'] = np.where(weather['precip'] > precip_threshold, 1, 0)
#     weather['humidity_high'] = np.where(weather['humidity'] > humidity_threshold, 1, 0)
#     weather['windspeed_high'] = np.where(weather['windspeed'] > windspeed_threshold, 1, 0)
#
#     # create the cloudy probability feature
#     weather['cloudy_prob'] = weather[(weather['cloudcover'] >= 0.7) & (weather['cloudcover'] <= 1)].shape[0] / \
#                              weather.shape[0]
#
#     return weather
#
#
# # load the weather data into a pandas DataFrame
# weather = pd.read_sql("select * from weather limit 1", conn)
#
# # Apply feature engineering
# weather = feature_engineering(weather)
#
# # Show the resulting data
# print(weather.head())
#
#
#
#
