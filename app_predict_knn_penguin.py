import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Load the penguin dataset (replace 'penguins.csv' with your data file)
@st.cache_data  # Cache the dataset for faster loading
def load_data():
    df = pd.read_csv('penguins.csv')
    return df

df = load_data()

# Preprocessing steps
def preprocess_data(df):
    # Separate features and target
    X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
    y = df['species']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define categorical and numerical features
    categorical_features = ['island', 'sex']
    numerical_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

    # Create transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
        ])

    return X_train, X_test, y_train, y_test, preprocessor

X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

# Train the KNN model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5)),  # You can adjust n_neighbors
])
model.fit(X_train, y_train)

# Streamlit app
st.title('Penguin Species Prediction')

# Input features
island = st.selectbox('Island', df['island'].unique())
bill_length_mm = st.number_input('Bill Length (mm)', min_value=0.0)
bill_depth_mm = st.number_input('Bill Depth (mm)', min_value=0.0)
flipper_length_mm = st.number_input('Flipper Length (mm)', min_value=0.0)
body_mass_g = st.number_input('Body Mass (g)', min_value=0.0)
sex = st.selectbox('Sex', df['sex'].unique())

# Create input data for prediction
input_data = pd.DataFrame([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]],
                         columns=['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex'])

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Species: **{prediction}**')
