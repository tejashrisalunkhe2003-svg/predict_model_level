import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
from streamlit_lottie import st_lottie

# Set page config for a professional look
st.set_page_config(page_title="AI Student Impact Predictor", page_icon="🎓", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4A90E2;
        color: white;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #ffffff;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function for Lottie Animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_unp87qum.json")

# Load the model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Header Section
st.title("🤖 AI Tool Impact Classifier")
st.write("Predict student outcomes based on AI usage patterns and demographics.")

with st.sidebar:
    st_lottie(lottie_ai, height=200, key="sidebar_anim")
    st.header("Input Features")
    st.info("Fill in the student details to get a prediction.")

# Form for user input
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        s_id = st.number_input("Student ID", min_value=0, step=1)
        age = st.number_input("Age", min_value=10, max_value=100, value=20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        edu = st.selectbox("Education Level", ["High School", "Undergraduate", "Postgraduate", "PhD"])
        city = st.text_input("City", "Boston")

    with col2:
        tool = st.selectbox("AI Tool Used", ["ChatGPT", "Claude", "Gemini", "Other"])
        usage = st.number_input("Daily Usage Hours", min_value=0.0, max_value=24.0, value=2.0)
        purpose = st.selectbox("Purpose", ["Study", "Research", "Coding", "General"])
        impact = st.selectbox("Impact on Grades", ["Improved", "Stayed Same", "Decreased"])

    submit = st.form_submit_button("Generate Prediction")

# Prediction Logic
if submit:
    # Create a DataFrame for the model
    # Note: KNN models usually require numerical data. 
    # If your model was trained on strings, this works. 
    # If not, you may need to apply a LabelEncoder here.
    features = pd.DataFrame([[
        s_id, age, gender, edu, city, tool, usage, purpose, impact
    ]], columns=[
        'Student_ID', 'Age', 'Gender', 'Education_Level', 'City', 
        'AI_Tool_Used', 'Daily_Usage_Hours', 'Purpose', 'Impact_on_Grades'
    ])

    try:
        prediction = model.predict(features)
        
        st.markdown("---")
        st.balloons()
        
        st.markdown(f"""
            <div class="prediction-box">
                <h2 style='color: #4A90E2;'>Prediction Result</h2>
                <h1 style='font-size: 50px;'>{prediction[0]}</h1>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.warning("Hint: Check if your model expects numerical values (Encoding) instead of raw text.")
