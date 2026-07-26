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


# Part 2: Mini-Project — Build the Weather Classifier
# Step 1: Fetch the Data

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 43.65,
    "longitude": -79.38,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)
print(df)

# Step 2: Engineer Labels

def label_running_day(row):
    """Return 1 if conditions are good for an outdoor run, 0 otherwise."""
    temp_ok    = 7 <= row["temperature_2m_max"] <= 26   # 45–79°F
    above_freeze = row["temperature_2m_min"] >= 0        # above freezing at dawn
    dry        = row["precipitation_sum"] < 3.0          # light rain or less
    not_windy  = row["wind_speed_10m_max"] < 30          # under 30 km/h
    return int(temp_ok and above_freeze and dry and not_windy)

df["good_for_running"] = df.apply(label_running_day, axis=1)

print(df["good_for_running"].value_counts())
print(f"\nFraction of good days: {df['good_for_running'].mean():.2f}")
# good_for_running
# 0    225 
# 1    140 
#Comments:
# That's about 62% of the year pretty reasonable for Toronto good for running 
# because no heavy rain, moderate temp and not too windy.

# Step 3: Train and Tune
X = df[["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
) 

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc")
grid.fit(X_train, y_train)
print(f"Best C: {grid.best_params_['clf__C']}")
print(f"Best CV AUC: {grid.best_score_:.3f}")

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))
y_probs_best = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_probs_best)
print(f"Test AUC: {test_auc:.3f}")

fpr, tpr, thresholds_scaled = roc_curve(y_test, y_probs_best)

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=test_auc).plot(ax=ax, name='Logistic Regression')

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
ax.set_title("ROC Curve")
ax.legend()
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/weather_roc.png")

# Step 4: Reflect on Evaluation
# AUC quality: The AUC of 0.90 is good, but lower than I initially expected.
# Since the "good running day" label follows exact rule-based cutoffs (temp,
# wind, rain), I assumed a near-perfect score was likely. The gap comes from
# a mismatch between the model and the label: Logistic Regression can only
# draw a straight line (linear decision boundary) to separate classes, but
# the actual rule is a "box" — four conditions that must all hold at once.
# A linear model structurally cannot represent that shape perfectly, which
# caps the achievable AUC even with clean, rule-based labels.

# Precision/recall: Looking at the classification report, precision for good
# days is high (0.90) but recall is lower (0.64). High precision means that
# when the model says "go run," it's usually right. Lower recall means it's
# missing a meaningful chunk of days that were actually good for running.
# So the model is conservative — it would rather stay silent than risk a
# bad recommendation.

# Error type: Since precision > recall here, the model's imbalance leans
# toward false negatives (missed good days) rather than false positives
# (wrongly recommending bad days). In other words, it under-recommends
# running rather than over-recommends it.

# Threshold choice: For a running app, these two error types aren't equally
# costly. Missing a good day (false negative) is a minor inconvenience — the
# user just doesn't get a recommendation they could have used. Recommending
# a bad day (false positive) is worse, since the user could show up to poor
# conditions based on the app's advice. Given that, the model's current
# caution at the default 0.5 threshold is reasonable. If we wanted to trade
# some of that caution for better coverage, we could lower the threshold to
# around 0.35-0.4, accepting more false positives in exchange for catching
# more of the good days we're currently missing.

# Step 5: Save the Model

os.makedirs('models', exist_ok=True)
joblib.dump(best_model, "models/weather_classifier.pkl")


metadata = {
    "python_version":  sys.version,
    "sklearn_version": sklearn.__version__,
    "features":        list(X_train.columns),
    "label":           "good_for_running",
    "label_thresholds": (
        "temp_max 7-26C, temp_min >= 0C, "
        "precipitation < 3.0mm, wind_speed < 30km/h"
    ),
    "best_params":     grid.best_params_,
    "test_auc":        round(test_auc, 4),
    "trained_on":      "2023 Open-Meteo, Toronto ON (lat 43.65, lon -79.38)",
    
}

print(metadata)

with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Model and metadata saved successfully:")
print(" - models/weather_classifier.pkl")
print(" - models/weather_classifier_metadata.json")