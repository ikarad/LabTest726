import streamlit as st
import numpy as np
import pickle
import pandas as pd

# โหลดโมเดลและตัวแปลงข้อมูล
with open('model_penguin_726.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# รับค่าจากผู้ใช้
culmen_length = st.sidebar.number_input('Culmen Length (mm)', value=40.0)
culmen_depth = st.sidebar.number_input('Culmen Depth (mm)', value=20.0)
flipper_length = st.sidebar.number_input('Flipper Length (mm)', value=190.0)
body_mass = st.sidebar.number_input('Body Mass (g)', value=3500.0)
island = st.sidebar.selectbox('Island', options=['Torgersen', 'Biscoe', 'Dream'])
sex = st.sidebar.selectbox('Sex', options=['MALE', 'FEMALE'])

# เตรียมข้อมูลผู้ใช้สำหรับการทำนาย
user_input = np.array([culmen_length, culmen_depth, flipper_length, body_mass, island, sex]).reshape(1, -1)

# แปลงข้อมูลของผู้ใช้
user_input[:, 4] = island_encoder.transform(user_input[:, 4])  # island encoding
user_input[:, 5] = sex_encoder.transform(user_input[:, 5])  # sex encoding
x_new =  pd.DataFrame() 
x_new['island'] = [island]
x_new['culmen_length_mm'] = [culmen_length]
x_new['culmen_depth_mm'] = [culmen_depth]
x_new['flipper_length_mm'] = [flipper_length]
x_new['body_mass_g'] = [body_mass]
x_new['sex'] = [sex]
x_new['island'] = island_encoder.transform(x_new['island'])
x_new['sex'] = sex_encoder.transform(x_new['sex'])

# ทำนายผล
prediction = model.predict(user_input)
# predicted_species = species_encoder.inverse_transform(prediction)

# แสดงผลการทำนาย
st.subheader('Prediction Result')
st.write(f"The predicted species is: {prediction}")
