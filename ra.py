import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the models
with open('decision_tree_model.pkl', 'rb') as file:
    clf_pipeline = pickle.load(file)

with open('random_forest.pkl', 'rb') as file:
    rf_pipeline = pickle.load(file)

# Load the dataset
df = pd.read_csv('dataset.csv')

# Fit the label encoder for the target variable
label_encoder = LabelEncoder()
df['Disease'] = label_encoder.fit_transform(df['Disease'])

# Load the symptom description data
df1 = pd.read_csv("symptom_Description.csv")
dfdict = df1.set_index('Disease').T.to_dict('list')

# Function to classify a new test case
def classify_new_case(model, new_case):
    new_case_df = pd.DataFrame([new_case])
    prediction = model.predict(new_case_df)
    disease = label_encoder.inverse_transform(prediction)
    return disease[0]

# Streamlit UI
st.title('Disease Prediction from Symptoms')
st.text('please enter atleast 3 symptoms')

# Create input fields for symptoms
symptoms = {}
all = st.text_input('Symptoms', 'none')
alls = all.split(',')
for i in range(1, 18):
    symptoms[f'Symptom_{i}'] = 'none'

for i in range(1, len(alls) + 1):
    symptoms[f'Symptom_{i}'] = alls[i - 1]

# Dropdown to choose model
model_choice = st.selectbox('Choose the model', ('Random Forest', 'Decision Tree'))

# Predict button
if st.button('Predict Disease'):
    if model_choice == 'Decision Tree':
        predicted_disease = classify_new_case(clf_pipeline, symptoms)
    else:
        predicted_disease = classify_new_case(rf_pipeline, symptoms)
    
    description = dfdict[predicted_disease][0]
    
    st.subheader('Predicted Disease')
    st.write(predicted_disease)
    
    st.subheader('Disease Description')
    st.write(description)
