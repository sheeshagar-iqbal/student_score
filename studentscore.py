import streamlit as st
import numpy as np
import joblib
import pandas as pd
import pickle

# --- Load the model ---
with open("score2.pkl", "rb") as f:
    loaded_model = pickle.load(f)
# Load model and scaler
# model = joblib.load("exam_score.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📊 Student Performance Prediction")

# Inputs
age = st.number_input("Age", min_value=10, max_value=25, value=18)
gender = st.selectbox("Gender", ["male", "female", "other"])
study_hours_per_day = st.slider("Study Hours per Day", 0, 8, 3)
social_media_hours = st.slider("Social Media Hours", 0, 8, 2)
netflix_hours = st.slider("Netflix & TV Hours", 0, 5, 1)
part_time_job = st.selectbox("Part-time Job", ["yes", "no"])
attendance_percentage = st.slider("Attendance (%)", 50, 100, 75)
sleep_hours = st.slider("Sleep Hours", 0, 8, 6)
diet_quality = st.selectbox("Diet Quality", ["poor", "good", "fair"])
exercise_frequency = st.slider("Exercise Frequency (day/week)", 0, 6, 3)
parental_education_level = st.selectbox("Parental Education Level", ["highschool", "bachelor", "master"])
internet_quality = st.selectbox("Internet Quality", ["poor", "average", "good"])
mental_health_rating = st.slider("Mental Health Rating (1-10)", 1, 10, 5)
extracurricular_participation = st.selectbox("Extracurricular Participation", ["yes", "no"])

# 🔹 Encoding using mapping dictionaries
gender_map = {"male": 1, "female": 0, "other": 2}
job_map = {"no": 0, "yes": 1}
diet_map = {"poor": 2, "good": 1, "fair": 0}
edu_map = {"highschool": 1, "bachelor": 0, "master": 2}
internet_map = {"poor": 2, "average": 0, "good": 1}
extra_map = {"no": 0, "yes": 1}

# Apply mappings
gender = gender_map[gender]
part_time_job = job_map[part_time_job]
diet_quality = diet_map[diet_quality]
parental_education_level = edu_map[parental_education_level]
internet_quality = internet_map[internet_quality]
extracurricular_participation = extra_map[extracurricular_participation]



# ['age', 'gender', 'study_hours_per_day', 'social_media_hours',
#        'netflix_hours', 'part_time_job', 'attendance_percentage',
#        'sleep_hours', 'diet_quality', 'exercise_frequency',
#        'parental_education_level', 'internet_quality', 'mental_health_rating',
#        'extracurricular_participation']
# Collect into array

data ={
    'age':age,
    'gender':gender,
    'study_hours_per_day':study_hours_per_day,
    'social_media_hours': social_media_hours,

    'netflix_hours':netflix_hours,
    'part_time_job':part_time_job,
    'attendance_percentage':attendance_percentage,

    'sleep_hours':sleep_hours,
    'diet_quality':diet_quality,
    'exercise_frequency':exercise_frequency,

    'parental_education_level':parental_education_level,
    'internet_quality':internet_quality,
    'mental_health_rating':mental_health_rating,
    'extracurricular_participation':extracurricular_participation
}
features = pd.DataFrame([data])

# Scale numeric
numeric_cols = ["age", "study_hours_per_day", "social_media_hours",
                "netflix_hours", "attendance_percentage",
                "sleep_hours", "exercise_frequency", "mental_health_rating"]

features[numeric_cols] = scaler.transform(features[numeric_cols])

# Predict
if st.button("Predict"):
    prediction = loaded_model.predict(features)
    prediction = np.clip(prediction, 0, 100)
    st.success(f"🎯 Predicted Output: {prediction[0]}")
