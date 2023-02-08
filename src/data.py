import psycopg2
import pandas as pd
import src.data as data
import csv
import re
import os
import glob
from sklearn.model_selection import train_test_split

con = psycopg2.connect(
    host="127.0.0.1",
    port="5432",
    database="postgres",
    user="admin4",
    password="admin4",
)


# Load train speed data
def fetch_rbc_train_speed_data_from_db():
    rbc_train_speed_query = pd.read_sql("select * from log_rbc_train", con)
    rbc_train_speed = pd.read_sql(rbc_train_speed_query, con)
    return rbc_train_speed


# Load weather data
def fetch_weather_data_from_db():
    weather_query = pd.read_sql("select * from weather", con)
    weather_df = pd.read_sql(weather_query, con)
    return weather_df


# Load LC data
def fetch_lc_data_from_db():
    lc_query = pd.read_sql("select * from log_lc_details", con)
    lc_df = pd.read_sql(lc_query, con)
    return lc_df


# Load level crossing movement time data
def fetch_lc_movement_data_from_db():
    lc_movement_data_query = """WITH cte AS (
SELECT
lc_id,
generated_timestamp_ts,
lc_message,
row_number() OVER (PARTITION BY lc_id ORDER BY generated_timestamp_ts) AS row_num
FROM (
SELECT
lc_id,
generated_timestamp AS generated_timestamp_ts,
lc_message
FROM log_lc_details

) AS subquery
),
cte2 AS (
WITH cte_2 AS (
SELECT
cte.lc_id,
cte.generated_timestamp_ts,
cte.lc_message,
row_number() OVER (PARTITION BY cte.lc_id ORDER BY cte.generated_timestamp_ts) AS row_num2
FROM cte
WHERE cte.lc_message = '1000'
)
SELECT
cte_2.lc_id,
cte_2.generated_timestamp_ts,
(
SELECT generated_timestamp_ts
FROM cte
WHERE lc_id = cte_2.lc_id
AND generated_timestamp_ts < cte_2.generated_timestamp_ts
AND lc_message <> '1000'
ORDER BY generated_timestamp_ts DESC
LIMIT 1
) AS prev_timestamp,
row_number() OVER (PARTITION BY cte_2.lc_id ORDER BY cte_2.generated_timestamp_ts) AS row_num2
FROM cte_2
)
SELECT
cte2.lc_id,
cte2.generated_timestamp_ts,
weather.windspeed,
weather.tempmin,
abs(extract(epoch from (LAG(cte2.generated_timestamp_ts) OVER (PARTITION BY cte2.lc_id ORDER BY cte2.generated_timestamp_ts) - cte2.prev_timestamp))) AS time_difference_2
FROM cte2
JOIN weather
ON date(cte2.generated_timestamp_ts) = date(weather.datetime)
ORDER BY cte2.lc_id, cte2.generated_timestamp_ts;"""
    lc_movement_data_df = pd.read_sql(lc_movement_data_query, con)
    return lc_movement_data_df


# Load point machine  data
def fetch_point_machine_data_from_db():
    point_machine_query = pd.read_sql("select * from log_point_movement", con)
    point_machine_query_df = pd.read_sql(point_machine_query, con)
    return point_machine_query_df


# Load point machine movement data
def fetch_point_machine_movement_data_from_db():
    point_machine_movement_query = pd.read_sql("""WITH move_data AS (
  SELECT point_machine_id, protocol_timestamp, 
         lead(protocol_timestamp) OVER (PARTITION BY point_machine_id ORDER BY protocol_timestamp) AS next_timestamp, 
         point_machine_state
  FROM log_point_movement
  WHERE point_machine_state = 'PT_MOVE' 
), state_data AS (
  SELECT point_machine_id, protocol_timestamp, point_machine_state
  FROM log_point_movement
  WHERE point_machine_state IN ('PT_RIGHT_SUPERV', 'PT_LEFT_SUPERV') 
)
SELECT point_machine_id, 
       avg(movement_time) as avg, 
      min(movement_time) as min, 
       max(movement_time) as max, 
       count(movement_time) as count
FROM (
  SELECT COALESCE(EXTRACT(EPOCH FROM min(state_data.protocol_timestamp - move_data.protocol_timestamp)),0) AS movement_time, 
         move_data.point_machine_id
  FROM move_data
  LEFT JOIN state_data ON move_data.point_machine_id = state_data.point_machine_id
    AND state_data.protocol_timestamp >= move_data.protocol_timestamp
    AND state_data.protocol_timestamp < move_data.next_timestamp
  GROUP BY move_data.protocol_timestamp, move_data.point_machine_id, state_data.protocol_timestamp
) AS subquery
WHERE movement_time < 100 and movement_time != 0
GROUP BY point_machine_id
ORDER BY point_machine_id
""", con)
    point_machine_movement_df = pd.read_sql(point_machine_movement_query, con)
    return point_machine_movement_df

def find_log_files(directory):
    log_files = []
    for file in glob.glob(f"{directory}/*.log*"):
        if file not in log_files:
            log_files.append(file)
    return log_files

directory = "/Users/adapasaikrishna/Documents/data_points/IM_LOGS"
log_files = find_log_files(directory)

# Create a list to store the values
rows = []
header = ['mceTimeStamp', 'rbcID', 'obuID', 'invokeID', 'trainNumber', 'trainLength', 'maxSpeed', 'airTight',
          'loadingGauge', 'tractions', 'availSTMs', 'rbcTrainCats', 'axleLoadCat', 'axleNumber', 'cDTrainCat',
          'oITrainCats', 'supportedNTCs', 'rbcTimeStamp']
rows.append(header)

# Iterate over each line in the log text file
for line in data:
    if "TRACTIONTYPE mainType" in line:
        # Extract the values using regular expressions
        m = re.search('mceTimeStamp="(\d+)"', line)
        if m:
            mceTimeStamp = m.group(1)
        m = re.search('rbcID="(\d+)"', line)
        if m:
            rbcID = m.group(1)
        m = re.search('obuID="(\d+)"', line)
        if m:
            obuID = m.group(1)
        m = re.search('invokeID="(\d+)"', line)
        if m:
            invokeID = m.group(1)
        m = re.search('trainNumber="(\d+)"', line)
        if m:
            trainNumber = m.group(1)
        m = re.search('trainLength="(\d+)"', line)
        if m:
            trainLength = m.group(1)
        m = re.search('maxSpeed="(\d+)"', line)
        if m:
            maxSpeed = m.group(1)
        m = re.search('airTight="(\w+)"', line)
        if m:
            airTight = m.group(1)
        m = re.search('loadingGauge="(\d+)"', line)
        if m:
            loadingGauge = m.group(1)
        m = re.search('tractions="(\w+)"', line)
        if m:
            tractions = m.group(1)
        m = re.search('availSTMs="(\w+)"', line)
        if m:
            availSTMs = m.group(1)
        m = re.search('rbcTrainCats="(\d+)"', line)
        if m:
            rbcTrainCats = m.group(1)
        m = re.search('axleLoadCat="(\w+)"', line)
        if m:
            axleLoadCat = m.group(1)
        m = re.search('axleNumber="(\d+)"', line)
        if m:
            axleNumber = m.group(1)
        m = re.search('cDTrainCat="(\d+)"', line)
        if m:
            cDTrainCat = m.group(1)
            m = re.search('oITrainCats="(\d+)"', line)
        if m:
            oITrainCats = m.group(1)
            m = re.search('supportedNTCs="(\d+)"', line)
        if m:
            supportedNTCs = m.group(1)
            m = re.search('rbcTimeStamp="(\d+)"', line)
        if m:
            rbcTimeStamp = m.group(1)

    #Creating a dataframe from extracted information

        train_df = pd.DataFrame({
        'mceTimeStamp': mceTimeStamp,
        'rbcID': rbcID,
        'obuID': obuID,
        'invokeID': invokeID,
        'trainNumber': trainNumber,
        'trainLength': trainLength,
        'maxSpeed': maxSpeed,
        'airTight': airTight,
        'loadingGauge': loadingGauge,
        'tractions': tractions,
        'availSTMs': availSTMs,
        'rbcTrainCats': rbcTrainCats,
        'axleLoadCat': axleLoadCat,
        'axleNumber': axleNumber,
        'cDTrainCat': cDTrainCat,
        'oITrainCats': oITrainCats,
        'supportedNTCs': supportedNTCs,
        'rbcTimeStamp': rbcTimeStamp
        }, index=[0])

    #Write the dataframe to a CSV file

        rbc_date = rbcTimeStamp[0:8]

        # Create the filename
        filename = f"{rbc_date}_rbc_log.csv"

        # Save the dataframe to a csv file
        train_df.to_csv(filename, index=False)


