
import streamlit as st
import pandas as pd
import joblib

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Mental Health Prediction",
    page_icon="☮️",
    layout="centered"
)

# --------------------------------
# LOAD TRAINED MODEL
# --------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/MindWell_AI_Model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please check MindWell_AI_Model.pkl.")
    st.stop()

# --------------------------------
# APPLICATION HEADER
# --------------------------------

st.title("Mental Health Prediction")
st.subheader("Mental Health Score Prediction")

st.write(
    "Enter your daily habits and lifestyle information "
    "to generate a predicted score using our machine learning model."
)

st.divider()

# --------------------------------
# USER INPUT FORM
# --------------------------------

with st.form("prediction_form"):

    st.header("Personal Information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=22
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    country = st.text_input(
        "Country",
        value="India"
    )

    academic_level = st.selectbox(
        "Academic Level",
        [
            "High School",
            "Undergraduate",
            "Graduate",
            "Postgraduate"
        ]
    )

    st.divider()

    st.header("Social Media Usage")

    platform = st.text_input(
        "Most Used Platform",
        value="Instagram"
    )

    purpose = st.text_input(
        "Purpose of Use",
        value="Entertainment"
    )

    daily_usage = st.number_input(
        "Average Daily Usage (Hours)",
        min_value=0.0,
        max_value=24.0,
        value=4.5,
        step=0.5
    )

    daily_unlocks = st.number_input(
        "Daily Phone Unlocks",
        min_value=0,
        max_value=2000,
        value=120
    )

    st.divider()

    st.header("Lifestyle Information")

    study_hours = st.number_input(
        "Daily Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=4.0,
        step=0.5
    )

    physical_activity = st.number_input(
        "Physical Activity Hours",
        min_value=0.0,
        max_value=24.0,
        value=1.5,
        step=0.5
    )

    sleep_hours = st.number_input(
        "Sleep Hours Per Night",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )

    submit = st.form_submit_button(
        "Predict Mental Health Score",
        use_container_width=True
    )

# --------------------------------
# PREDICTION
# --------------------------------

if submit:

    if not country.strip() or not platform.strip() or not purpose.strip():
        st.error("Please fill in all categorical fields.")
        st.stop()

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Country": country.strip(),
        "Academic_Level": academic_level,
        "Most_Used_Platform": platform.strip(),
        "Purpose_Of_Use": purpose.strip(),
        "Avg_Daily_Usage_Hours": daily_usage,
        "Daily_Unlocks": daily_unlocks,
        "Study_Hours": study_hours,
        "Physical_Activity_Hours": physical_activity,
        "Sleep_Hours_Per_Night": sleep_hours
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Mental Health Score",
            value=f"{prediction:.2f} / 10"
        )

        st.success("Prediction generated successfully!")

        with st.expander("View submitted information"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --------------------------------
# DISCLAIMER
# --------------------------------

st.divider()

st.caption(
    "Disclaimer: MindWell AI is an educational machine learning "
    "project. Its predictions are based on patterns in the training "
    "dataset and are not clinical assessments, diagnoses, or "
    "substitutes for professional mental health advice."
)
