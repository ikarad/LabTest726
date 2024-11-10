import streamlit as st
import numpy as np
import pickle

# โหลดโมเดลและตัวแปลงข้อมูล
with open('model_penguin_726.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# รับค่าจากผู้ใช้
culmen_length = st.sidebar.number_input('Culmen Length (mm)', min_value=0.0, max_value=100.0, value=40.0)
culmen_depth = st.sidebar.number_input('Culmen Depth (mm)', min_value=0.0, max_value=100.0, value=20.0)
flipper_length = st.sidebar.number_input('Flipper Length (mm)', min_value=0.0, max_value=100.0, value=190.0)
body_mass = st.sidebar.number_input('Body Mass (g)', min_value=0.0, max_value=10000.0, value=3500.0)
island = st.sidebar.selectbox('Island', options=['Torgersen', 'Biscoe', 'Dream'])
sex = st.sidebar.selectbox('Sex', options=['MALE', 'FEMALE'])

# เตรียมข้อมูลผู้ใช้สำหรับการทำนาย
user_input = np.array([culmen_length, culmen_depth, flipper_length, body_mass, island, sex]).reshape(1, -1)

# แปลงข้อมูลของผู้ใช้
user_input[:, 4] = island_encoder.transform(user_input[:, 4])  # island encoding
user_input[:, 5] = sex_encoder.transform(user_input[:, 5])  # sex encoding

# ทำนายผล
prediction = model.predict(user_input)
predicted_species = species_encoder.inverse_transform(prediction)

# แสดงผลการทำนาย
st.subheader('Prediction Result')
st.write(f"The predicted species is: {predicted_species[0]}")
