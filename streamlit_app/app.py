import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. SET UP PATHS & LOAD ASSETS ---
# Uses absolute pathing to prevent "FileNotFound" regardless of where streamlit is run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

@st.cache_resource
def load_assets():
    """Loads all 4 saved joblib files from the models directory."""
    binary = joblib.load(os.path.join(MODEL_PATH, 'binary_model.joblib'))
    multi = joblib.load(os.path.join(MODEL_PATH, 'multiclass_model.joblib'))
    reg = joblib.load(os.path.join(MODEL_PATH, 'regression_model.joblib'))
    scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.joblib'))
    return binary, multi, reg, scaler

try:
    binary_model, multi_model, reg_model, scaler = load_assets()
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.info("Ensure all 4 .joblib files are in the 'models' folder.")
    st.stop()

# --- 2. USER INTERFACE ---
st.set_page_config(page_title="Diabetes Health AI", layout="wide")
st.title("🩺 Diabetes Health Indicator Portal")
st.markdown("---")

# Layout: 2 columns (Inputs on left, Predictions on right)
col_input, col_pred = st.columns([1, 1.5], gap="large")

with col_input:
    st.header("Patient Vital Signs")
    
    # These are the primary features users usually know
    age = st.slider("Age", 18, 100, 45)
    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0)
    glucose_f = st.number_input("Fasting Glucose (mg/dL)", 50, 300, 100)
    hba1c = st.slider("HbA1c Level (%)", 4.0, 15.0, 5.5)
    
    with st.expander("Additional Health Factors"):
        smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Smoker" if x==1 else "Non-Smoker")
        exercise = st.number_input("Exercise (mins/week)", 0, 1000, 150)
        fam_history = st.selectbox("Family History of Diabetes?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        systolic = st.number_input("Systolic BP", 80, 200, 120)
        diastolic = st.number_input("Diastolic BP", 50, 120, 80)

# --- 3. DATA PROCESSING ---
# We must create a dictionary with ALL 28 columns in the EXACT order the scaler expects
input_data = {
    'age': age, 
    'gender': 0, 'ethnicity': 0, 'education_level': 2, 'income_level': 3, 'employment_status': 1, 
    'smoking_status': smoking, 
    'alcohol_consumption_per_week': 2,
    'physical_activity_minutes_per_week': exercise, 
    'diet_score': 7, 'sleep_hours_per_day': 7, 'screen_time_hours_per_day': 3, 
    'family_history_diabetes': fam_history,
    'hypertension_history': 1 if systolic > 140 else 0, 
    'cardiovascular_history': 0, 
    'bmi': bmi,
    'waist_to_hip_ratio': 0.9, 
    'systolic_bp': systolic, 
    'diastolic_bp': diastolic, 
    'heart_rate': 72,
    'cholesterol_total': 200, 'hdl_cholesterol': 50, 'ldl_cholesterol': 100, 'triglycerides': 150, 
    'glucose_fasting': glucose_f, 
    'glucose_postprandial': glucose_f + 40, 
    'insulin_level': 15, 
    'hba1c': hba1c
}

# The sequence must match scaler.feature_names_in_
expected_order = [
    'age', 'gender', 'ethnicity', 'education_level', 'income_level', 'employment_status', 
    'smoking_status', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'family_history_diabetes', 
    'hypertension_history', 'cardiovascular_history', 'bmi', 'waist_to_hip_ratio', 
    'systolic_bp', 'diastolic_bp', 'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 
    'ldl_cholesterol', 'triglycerides', 'glucose_fasting', 'glucose_postprandial', 
    'insulin_level', 'hba1c'
]

features_df = pd.DataFrame([input_data])[expected_order]

# --- 4. PREDICTIONS ---
with col_pred:
    st.header("Clinical Assessment")
    
    try:
        # Step A: Scale the data
        features_scaled = scaler.transform(features_df)

        # Step B: Run all three machines
        binary_pred = binary_model.predict(features_scaled)[0]
        binary_proba = binary_model.predict_proba(features_scaled)[0][1]
        
        multi_stage = multi_model.predict(features_scaled)[0]
        
        risk_score = reg_model.predict(features_scaled)[0]

        # Step C: Display Results in Tabs
        tab1, tab2, tab3 = st.tabs(["Binary Diagnosis", "Staging", "Risk Analytics"])
        
        with tab1:
            result_text = "DIABETIC" if binary_pred == 1 else "NON-DIABETIC"
            color = "red" if binary_pred == 1 else "green"
            st.subheader(f"Status: :{color}[{result_text}]")
            st.metric("Confidence Level", f"{binary_proba*100:.1f}%")
            st.progress(binary_proba)

        with tab2:
            st.subheader(f"Estimated Diabetes Stage: {multi_stage}")
            
            if multi_stage == 0:
              st.success("Stage 0: Healthy / Low Risk")
            elif multi_stage == 1:
              st.info("Stage 1: Pre-diabetic")
            elif multi_stage <= 3:
              st.warning("Stage 2-3: Early-Stage Diabetes")
            else:
             st.error(f"Stage {multi_stage}: Advanced / Chronic Diabetes")
        
             st.info("Clinical Note: Higher stages indicate increased complexity and potential complications.")

        with tab3:
            st.subheader("Diabetes Risk Score")
            st.metric("Score (0-100)", f"{risk_score:.2f}")
            if risk_score > 70:
                st.warning("⚠️ High Risk: Immediate clinical consultation advised.")
            elif risk_score > 40:
                st.info("Moderate Risk: Lifestyle changes recommended.")
            else:
                st.success("Low Risk: Maintain current health habits.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")