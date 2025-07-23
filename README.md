import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
# Basic Streamlit app config
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="wide")
# Simulated dataset 
sample_salary_data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Education': ["B.Tech", "M.Tech", "B.Sc", "MBA", "B.Com", 
                  "M.Sc", "BBA", "PhD", "M.Tech", "B.Tech"],
    'Job_Role': ["Software Engineer", "Data Analyst", "Web Developer", 
                 "Project Manager", "Marketing Executive",
                 "System Admin", "HR Manager", "Data Scientist", 
                 "Business Analyst", "Junior Developer"],
    'Level': ["Entry", "Junior", "Mid", "Senior", "Lead", 
              "Senior", "Junior", "Mid", "Senior", "Entry"],
    'Location': ["Delhi", "Mumbai", "Bangalore", "Chennai", "Hyderabad",
                 "Pune", "Ahmedabad", "Kolkata", "Jaipur", "Remote"],
    'Salary': [40000, 60000, 50000, 80000, 70000, 
               90000, 75000, 120000, 110000, 30000]}
df = pd.DataFrame(sample_salary_data)
label_maps = {}
for feature in ['Education', 'Job_Role', 'Level', 'Location']:
    encoder = LabelEncoder()
    df[f'{feature}_Encoded'] = encoder.fit_transform(df[feature])
    label_maps[feature] = encoder  

features_to_use = ['Experience', 'Education_Encoded', 'Job_Role_Encoded', 'Level_Encoded', 'Location_Encoded']
X_train = df[features_to_use]
y_train = df['Salary']
regressor = LinearRegression()
regressor.fit(X_train, y_train)  # Fit once on this mini sample data
# === Streamlit UI starts here ===
st.title("Indian Salary Predictor")
st.write("Curious what your salary *might* look like? Feed in some details and we’ll take a guess.")
years_exp = st.slider("Years of Experience", min_value=0, max_value=30, value=2)
edu_level = st.selectbox("Highest Education", sorted(df['Education'].unique()))
job_title = st.selectbox("Job Role", sorted(df['Job_Role'].unique()))
job_level = st.selectbox("Seniority Level", sorted(df['Level'].unique()))
job_location = st.selectbox("Work Location", sorted(df['Location'].unique()))

def get_salary_prediction():
    try:
        encoded_inputs = {
            'Experience': [years_exp],
            'Education_Encoded': [label_maps['Education'].transform([edu_level])[0]],
            'Job_Role_Encoded': [label_maps['Job_Role'].transform([job_title])[0]],
            'Level_Encoded': [label_maps['Level'].transform([job_level])[0]],
            'Location_Encoded': [label_maps['Location'].transform([job_location])[0]] }
        input_row = pd.DataFrame(encoded_inputs)
        predicted_salary = regressor.predict(input_row)[0]
        return round(predicted_salary, 2)
    except Exception as error:
        st.error("Hmm, something went off while predicting:")
        st.code(str(error))
        return None
if st.button("Predict Salary"):
    estimated_salary = get_salary_prediction()
    if estimated_salary:
        st.success(f"Estimated Salary: ₹{estimated_salary:,.2f} per year")
if st.checkbox("Show training sample data"):
    st.dataframe(df[['Experience', 'Education', 'Job_Role', 'Level', 'Location', 'Salary']])




