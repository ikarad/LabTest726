
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to load and prepare the data
def load_data():
    # Sample Data - Replace with your actual dataset
    df = pd.read_csv("penguins_size.csv")  # Make sure you have the penguin dataset
    return df

# Function to encode the categorical features
def encode_features(df):
    species_encoder = LabelEncoder()
    island_encoder = LabelEncoder()
    sex_encoder = LabelEncoder()

    df['species'] = species_encoder.fit_transform(df['species'])
    df['island'] = island_encoder.fit_transform(df['island'])
    df['sex'] = sex_encoder.fit_transform(df['sex'])
    
    return df, species_encoder, island_encoder, sex_encoder

# Function to train the KNN model
def train_knn_model(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    return model

# Load and prepare data
df = load_data()

# Preprocess the data
df, species_encoder, island_encoder, sex_encoder = encode_features(df)

# Prepare training and testing sets
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = train_knn_model(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Streamlit interface
st.title('Penguin Species Prediction with KNN')
st.write("This app predicts the species of penguins based on their physical attributes.")

# Collect user input for prediction
st.sidebar.header('Input Features')

culmen_length = st.sidebar.number_input('Culmen Length (mm)', min_value=0.0, max_value=100.0, value=40.0)
culmen_depth = st.sidebar.number_input('Culmen Depth (mm)', min_value=0.0, max_value=100.0, value=20.0)
flipper_length = st.sidebar.number_input('Flipper Length (mm)', min_value=0.0, max_value=100.0, value=190.0)
body_mass = st.sidebar.number_input('Body Mass (g)', min_value=0.0, max_value=10000.0, value=3500.0)
island = st.sidebar.selectbox('Island', options=['Torgersen', 'Biscoe', 'Dream'])
sex = st.sidebar.selectbox('Sex', options=['MALE', 'FEMALE'])

# Prepare the user input for prediction
user_input = np.array([culmen_length, culmen_depth, flipper_length, body_mass, island, sex]).reshape(1, -1)

# Apply label encoding to the user input
user_input[:, 4] = island_encoder.transform(user_input[:, 4])  # island encoding
user_input[:, 5] = sex_encoder.transform(user_input[:, 5])  # sex encoding

# Make prediction
prediction = model.predict(user_input)
predicted_species = species_encoder.inverse_transform(prediction)

# Display the result
st.subheader('Prediction Result')
st.write(f"The predicted species is: {predicted_species[0]}")

# Display model accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")
