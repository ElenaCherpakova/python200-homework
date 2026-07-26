import os
import requests
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
)
import joblib

# Task 1: Load and Verify

loaded_pipline = joblib.load("models/weather_classifier.pkl")
with open("models/weather_classifier_metadata.json", "r") as f:
    loaded_metadata = json.load(f)
    print(loaded_metadata['trained_on'])
    print(loaded_metadata['features'])
    print(loaded_metadata['test_auc'])

# Task 2: Predict on New Data

new_days = pd.DataFrame([
    [20.0,  12.0, 0.0,  10.0],  
    [24.0,  16.0, 0.5,  15.0],   
    [-5.0, -12.0, 0.0,  20.0],   
    [22.0,  14.0, 25.0, 18.0],   
    [ 7.1,   0.5, 2.9,  29.0], 
], columns=loaded_metadata["features"])

predictions = loaded_pipline.predict(new_days)
probabilities = loaded_pipline.predict_proba(new_days)[:, 1]

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    row = new_days.iloc[i]
    label = "good" if pred == 1 else "skip"
    print(f"Day {i+1}: temp_max={row['temperature_2m_max']}, "
        f"temp_min={row['temperature_2m_min']}, "
        f"precip={row['precipitation_sum']}, "
        f"wind={row['wind_speed_10m_max']} "
        f"-> predicted: {label}, confidence: {prob:.2f}")
    
# Day 1: temp_max=20.0, temp_min=12.0, precip=0.0, wind=10.0 -> predicted: good, confidence: 0.81
# Day 2: temp_max=24.0, temp_min=16.0, precip=0.5, wind=15.0 -> predicted: good, confidence: 0.82
# Day 3: temp_max=-5.0, temp_min=-12.0, precip=0.0, wind=20.0 -> predicted: skip, confidence: 0.07
# Day 4: temp_max=22.0, temp_min=14.0, precip=25.0, wind=18.0 -> predicted: skip, confidence: 0.00
# Day 5: temp_max=7.1, temp_min=0.5, precip=2.9, wind=29.0 -> predicted: skip, confidence: 0.03

# Task 3: Reflect
# Q1: My borderline day got confidence=0.03, which is far from 0.5, so
# the model was confident but wrong -- it learned that low temp + high
# wind + some rain together usually meant a bad day, even though my rule
# said this exact combo counted as "good," and if I saw a real 0.52 I'd
# just tell the user it's borderline instead of a confident yes/no.

# Q2: Running predict_weather.py first throws a FileNotFoundError since
# the .pkl doesn't exist yet -- wrapping the load in try/except and
# printing "run train_weather_classifier.py first" instead of a raw
# traceback would make it clear what to do.

# Q3: predict_weather.py would need to call Open-Meteo's forecast API
# (not archive-api) to get tomorrow's predicted temp_max, temp_min,
# precip, and wind instead of hardcoded hypothetical values, then feed
# those into the already-trained model with predict()/predict_proba()
# the model itself stays frozen and is never retrained here, only
# the input data changes daily. I'd also wrap the API call in
# try/except in case the API is unavailable, and log the date and
# result of each run so there's a record of what was predicted and when.