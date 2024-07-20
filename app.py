import streamlit as st
import numpy as np
import pickle
from datetime import datetime,date
import pandas as pd

st.set_page_config(layout='wide')
st.write("""
<div style='text-align:center'>
    <h1 style='color:#5e17eb;'>Energy Consumption and Prediction</h1>
</div>
""", unsafe_allow_html=True)
st.write('')
st.write('')
st.write('')
df=pd.read_csv('energy_consumption.csv')
primary_use_list = df['primary_use'].unique()
# Convert string to datetime object
min_date_str = '2016-01-01 00:00:00'
min_date = datetime.strptime(min_date_str, '%Y-%m-%d %H:%M:%S')
max_date_str = '2016-12-31 23:00:00'
max_date = datetime.strptime(max_date_str, '%Y-%m-%d %H:%M:%S')
min_squft = min(df['square_feet'])
max_squft = max(df['square_feet'])
min_built = min(df['year_built'])
max_built = max(df['year_built'])
min_air = min(df['air_temperature'])
max_air = max(df['air_temperature'])
min_dew = min(df['dew_temperature'])
max_dew = max(df['dew_temperature'])
min_precip =min(df['precip_depth_1_hr'])
max_percip = max(df['precip_depth_1_hr'])
min_sea = min(df['sea_level_pressure'])
max_sea = max(df['sea_level_pressure'])
min_direction = min(df['wind_direction'])
max_direction =max(df['wind_direction'])
min_speed = min(df['wind_speed'])
max_speed = max(df['wind_speed'])
# get input from users
with st.form('Classification'):
    col1, col2, col3 = st.columns([0.5, 0.1, 0.5])
    with col1:
        building_id = st.number_input(label='Building_Id (Min: 1 & Max: 100)',min_value=1,max_value=100,value=1,key='building_id')
        primary_use = st.selectbox(label='Primary use', options=primary_use_list)
        date = st.date_input(label='Date', min_value=min_date.date(), max_value=max_date.date(), value=min_date.date())
        time_str = st.text_input(label='Time (HH:MM:SS)', value=min_date.strftime('%H:%M:%S'))
        try:
            datetime_str = f"{date} {time_str}"
            timestamp = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

        except ValueError:
            st.error("Incorrect time format, please use HH:MM:SS")
        square_feet = st.number_input(label=f'square feet (min: {min_squft} & max: {max_squft})',min_value=min_squft,max_value=max_squft,value=min_squft,key = 'square_feet')
        year_built = st.number_input(label=f'year_built (min: {min_built} & max: {max_built})',min_value=min_built,max_value=max_built,value=max_built,key='year_built')
    with col3:
        air_temperature = st.number_input(label=f'air_temperature (min: {min_air} & max: {max_air})',min_value=min_air,max_value=max_air,value=min_air,key='air_temperature')
        dew_temperature = st.number_input(label=f'dew_temperature (min: {min_dew} & max: {max_dew})',min_value=min_dew,max_value=max_dew,value=min_dew,key='dew_temperature')
        precip_depth_1_hr = st.number_input(label=f'precip_depth_1_hr(min: 0.0 & max: {max_percip})',min_value=0.0,max_value=max_percip,value=0.0,key='precip_depth_1_hr')
        sea_level_pressure = st.number_input(label=f'sea_level_pressure (min: {min_sea} & max: {max_sea})',min_value=min_sea,max_value=max_sea,value=min_sea,key='sea_level_pressure')
        wind_direction = st.number_input(label=f'wind_direction (min: {min_direction} & max: {max_direction})',min_value=min_direction,max_value=max_direction,value=min_direction,key='wind_direction')
        wind_speed = st.number_input(label=f'wind_speed(min: {min_speed} & max: {max_speed})',min_value=min_speed,max_value=max_speed,value=min_speed,key='wind_speed')
        st.write('')
        st.write('')
        st.write('')
    button = st.form_submit_button(label='SUBMIT')
    if button:
        # load the regression pickle model
        with open(r'regression_model.pkl', 'rb') as f:
            model = pickle.load(f)

        primary_use_map = {'Entertainment/public assembly': 0, 'Lodging/residential': 1, 'Office': 2, 'Other': 3,
                           'Parking': 4, 'Retail': 5, 'Education': 6}

        # make array for all user input values in required order for model prediction
        user_data = np.array([[building_id,
                               primary_use_map[primary_use],
                               np.log1p(square_feet),
                               year_built,
                               air_temperature,
                               np.square(dew_temperature),
                               np.log1p(precip_depth_1_hr),
                               sea_level_pressure,
                               wind_direction,
                               wind_speed , timestamp.month, timestamp.day, timestamp.hour
                               ]])

        # model predict the selling price based on user input
        y_pred = model.predict(user_data)


        meter_reading = y_pred[0]

        # round the value with 2 decimal point (Eg: 1.35678 to 1.36)
        meter_reading = round(meter_reading, 2)
        st.write(f'Predicted Meter reading is : {meter_reading}')








