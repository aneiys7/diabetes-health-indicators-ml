from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import joblib
import os

def clean_and_scale(df, is_training=True, scaler_path='../models/scaler.joblib'):
    # 1. Handling Missing Values
    imputer = SimpleImputer(strategy='median')
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # 2. Label Encoding
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # 3. Standardizing
    targets = ['diagnosed_diabetes', 'diabetes_stage', 'diabetes_risk_score']
    feature_cols = [col for col in df.columns if col not in targets]
    
    if is_training:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        # Save the scaler so the App can use it later!
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        return df, scaler
    else:
        # Load the ALREADY TRAINED scaler for the App
        scaler = joblib.load(scaler_path)
        df[feature_cols] = scaler.transform(df[feature_cols]) # Notice: .transform, NOT .fit_transform
        return df

