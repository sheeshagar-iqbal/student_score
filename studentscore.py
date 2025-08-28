import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("exam_score.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📊 Student Performance Prediction")

# Inputs
age = st.number_input("Age", min_value=10, max_value=30, value=18)
gender = st.selectbox("Gender", ["male", "female", "other"])
study_hours = st.slider("Study Hours per Day", 0, 12, 3)
social_media = st.slider("Social Media Hours", 0, 12, 2)
netflix = st.slider("Netflix Hours and TV Hours", 0, 12, 1)
part_time_job = st.selectbox("Part-time Job", ["yes", "no"])
attendance = st.slider("Attendance (%)", 0, 100, 75)
sleep = st.slider("Sleep Hours", 0, 12, 7)
diet = st.selectbox("Diet Quality", ["poor", "good", "fair"])
exercise = st.slider("Exercise Frequency (days/week)", 0, 6, 3)
parent_edu = st.selectbox("Parental Education Level", ["highschool", "bachelor", "master"])
internet = st.selectbox("Internet Quality", ["poor", "average", "good"])
mental_health = st.slider("Mental Health Rating (1-10)", 1, 10, 5)
extracurricular = st.selectbox("Extracurricular Participation", ["yes", "no"])

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
diet = diet_map[diet]
parent_edu = edu_map[parent_edu]
internet = internet_map[internet]
extracurricular = extra_map[extracurricular]



# ['age', 'gender', 'study_hours_per_day', 'social_media_hours',
#        'netflix_hours', 'part_time_job', 'attendance_percentage',
#        'sleep_hours', 'diet_quality', 'exercise_frequency',
#        'parental_education_level', 'internet_quality', 'mental_health_rating',
#        'extracurricular_participation']
# Collect into array
features = np.array([[age, gender, study_hours, social_media, netflix,
                      part_time_job, attendance, sleep, diet, exercise,
                      parent_edu, internet, mental_health, extracurricular]])

# Scale numeric
features = scaler.fit_transform(features)

# Predict
if st.button("Predict"):
    prediction = model.predict(features)
    st.success(f"🎯 Predicted Output: {prediction[0]}")
