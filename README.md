# diabetes-health-indicators-ml
https://diabetes-health-indicators-ml-qxyejzkezh3fadruc7gv66.streamlit.app/

🩺 Diabetes Health AI Portal
A comprehensive machine learning deployment that utilizes 28 clinical and lifestyle indicators to provide a three-tier health assessment. This project leverages three distinct models to provide binary diagnosis, disease staging, and numerical risk scoring.
 
 Live Streamlit App


 Project Capabilities

The portal addresses three specific prediction tasks as outlined in the project requirements:

Binary Classification: Determines the likelihood of a "Diabetic" vs "Non-Diabetic" status with real-time confidence scores.

Multiclass Staging: Categorizes the progression of the condition into 5 distinct stages (Stage 0: Healthy to Stage 4: Advanced/Chronic).

Risk Regression: Calculates a normalized Diabetes Risk Score (0-100) to quantify health vulnerability.


Technical Stack & Architecture

Preprocessing: StandardScaler pipeline to normalize 28 features including BMI, HbA1c, and Fasting Glucose.

Models: Serialized .joblib models (Binary, Multiclass, and Regression).

Frontend: Streamlit-based UI with interactive sliders, expandable health forms, and tabbed analytics.

Deployment: Fully CI/CD integrated via GitHub and Streamlit Community Cloud.


Key Insights

Through Exploratory Data Analysis (EDA), the following indicators were identified as the strongest predictors for high-risk assessments:

HbA1c Level: The most significant long-term blood sugar marker.

Fasting Glucose: Primary indicator for acute diabetic status.

BMI & Age: Critical demographic and physical factors influencing the risk score.