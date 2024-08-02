import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model_path = 'C:/Users/AshokKumar/OneDrive/Desktop/Logistic/logistic_regression_model.pkl'
scaler_path = 'C:/Users/AshokKumar/OneDrive/Desktop/Logistic/scaler.pkl'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Set up the Streamlit app
st.title('Titanic Survival Prediction')

# Input fields for the user
age = st.number_input('Age')
pclass = st.number_input('Pclass', min_value=1, max_value=3)
sibsp = st.number_input('Number of Siblings/Spouses Aboard')
parch = st.number_input('Number of Parents/Children Aboard')
fare = st.number_input('Fare')
sex = st.selectbox('Sex', ['male', 'female'])

# Prepare the input data for prediction
input_data = {
    'Age': [age],
    'Pclass': [pclass],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Sex_female': [1 if sex == 'female' else 0]
}
input_df = pd.DataFrame(input_data)

# Ensure the columns match the training data format
# Drop the 'Sex_male' column to match the dummy encoding drop_first=True
input_df = input_df.reindex(columns=['Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Sex_female'], fill_value=0)

# Scale the input data
input_df_scaled = scaler.transform(input_df)

# Predict the result
prediction = model.predict(input_df_scaled)
prediction_proba = model.predict_proba(input_df_scaled)

# Display results
st.write(f'Prediction: {"Survived" if prediction[0] == 1 else "Not Survived"}')
st.write(f'Probability of Survival: {prediction_proba[0][1]:.2f}')
st.write(f'Probability of Not Surviving: {prediction_proba[0][0]:.2f}')
