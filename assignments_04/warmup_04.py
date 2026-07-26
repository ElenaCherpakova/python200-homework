# --- ROC and AUC ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
    confusion_matrix

)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Q1

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

knc = KNeighborsClassifier(n_neighbors=5)
knc.fit(X_train_scaled, y_train)

y_probs_raw = clf.predict_proba(X_test)[:, 1]
fpr_raw, tpr_raw, thresholds_raw = roc_curve(y_test, y_probs_raw)
y_probs_scaled = knc.predict_proba(X_test_scaled)[:, 1]
fpr_scaled, tpr_scaled, thresholds_scaled = roc_curve(y_test, y_probs_scaled)

auc_raw = roc_auc_score(y_test, y_probs_raw)
auc_scaled = roc_auc_score(y_test, y_probs_scaled)
print(f"AUC score for raw: {auc_raw:.3f}")
print(f"AUC score for scaled: {auc_scaled:.3f}")

#Comments:
# AUC score for raw: 0.706
# AUC score for KNN (0.939.) is higher compared to logistic regression.
# KNN separates the 2 classes better than logistic regression, no matter 
# what threshold we pick.

# Q2

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=fpr_raw, tpr=tpr_raw, roc_auc=auc_raw).plot(ax=ax, name='Logistic Regression')
RocCurveDisplay(fpr=fpr_scaled, tpr=tpr_scaled, roc_auc=auc_scaled).plot(ax=ax, name='KNN')

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
ax.set_title("ROC Curve")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/roc_comparison.png")
plt.show()

#finding FPR where TPR first reaches 0.80 on each curve
fpr_raw_at_80 = fpr_raw[np.argmax(tpr_raw >= 0.80)]
fpr_scaled_at_80 = fpr_scaled[np.argmax(tpr_scaled >= 0.80)]
print(f"FPR at TPR=0.80 - Logistic Regression: {fpr_raw_at_80:.3f}")
print(f"FPR at TPR=0.80 - KNN: {fpr_scaled_at_80:.3f}")
# Comments:
# FPR at TPR=0.80 - Logistic Regression: 0.580
# FPR at TPR=0.80 - KNN: 0.110
# Logistic Regression has the lower FPR at TPR=0.80.
# which means if we need to catch 80% of positives, Logistic Regression gives
# fewer false alarms than KNN at that point on the curve.

# Q3
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
y_probs_lr = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs_lr)

best_f1 = 0
best_threshold = None
best_idx = None
for i, threshold in enumerate(thresholds):
    y_pred = (y_probs_lr >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_idx = i

y_pred_default = (y_probs_lr >= 0.5).astype(int)
f1_default = f1_score(y_test, y_pred_default)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_default).ravel()
tpr_default = tp / (tp + fn)
fpr_default = fp / (fp + tn)
print(f"Best threshold: {best_threshold:.3f}")
print(f"TPR: {tpr[best_idx]}")
print(f"FPR: {fpr[best_idx]}")
print(f"F1: {best_f1}")
print()
print(f"Default threshold: 0.500")
print(f"TPR: {tpr_default:.3f}")
print(f"FPR: {fpr_default:.3f}")
print(f"F1: {f1_default:.3f}")
#Comments:
# The optimal threshold (0.276) is lower than default (0.5), so the model
# predicts positive more easily. This raises TPR from 0.620 to 0.890 (more
# true positives caught) but also raises FPR from 0.300 to 0.690 (many more
# false alarms). F1 barely improves (0.646 -> 0.690) since the gain from
# extra true positives is mostly cancelled out by all the extra false positives.
# In real app we would pick a threshold below 0.5 when missing a possitive case 
# is way more costly that a false alarm

# --- GridSearchCV ---
# Q1
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

param_grid = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc")
grid.fit(X_train, y_train)
best_pipe = grid.best_estimator_

print(f"Best C: {grid.best_params_['clf__C']}")
print(f"Best CV AUC: {grid.best_score_:.3f}")

y_pred = best_pipe.predict(X_test)
print(classification_report(y_test, y_pred))

y_probs_best = best_pipe.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_probs_best)
print(f"Test AUC: {test_auc:.3f}")

# print(pd.DataFrame(grid.cv_results_).columns.tolist())
results = pd.DataFrame(grid.cv_results_)[["param_clf__C", "mean_test_score", "std_test_score"]]
results = results.sort_values("mean_test_score", ascending=False)
print(results.to_string(index=False))

# Comments:
# Best C: 100.0 (default is C=1.0)
# grid search picked a different C than default, but CV scores are almost
# the same (0.7725 vs 0.7727) - barely any difference.
# test AUC is 0.706, same as default C=1.0 from Q1. so grid search didn't
# really improve anything here - C doesn't matter much for this data.

# Q2
print("--------Q2--------")
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

param_grid = {
    "classifier__max_depth": [2, 3, 5, 8, None]
}

grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring="roc_auc")
grid.fit(X_train, y_train)

best_tree_pipe = grid.best_estimator_
y_probs_best = best_tree_pipe.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_probs_best)
print(f"Test AUC: {test_auc:.3f}")
print(f"Best max_depth: {grid.best_params_['classifier__max_depth']}")
print(f"Best CV AUC: {grid.best_score_:.3f}")

# Comments:
# Best max_depth: 5
# Best CV AUC: 0.917
# Test AUC: 0.935
# decision tree (0.935) has way higher AUC than logistic regression from
# Q1 (0.706). so on AUC alone, decision tree wins by a lot.
# but AUC is not the only thing to look at. decision tree with no max_depth
# limit (None) actually did worse (0.863) and less stable (std 0.039) than
# max_depth=5 - that's a sign of overfitting when the tree gets too deep.
# logistic regression is simpler, more interpretable, and less likely to
# overfit, even if its AUC is lower.
# for further development, I'd bring the decision tree (with max_depth=5)
# forward since it's clearly better here - but I'd also keep an eye on
# overfitting and maybe check other things like training time and how easy
# the model is to explain to others before deciding for real.

# Q3
print("--------Q3--------")
results = pd.DataFrame(grid.cv_results_)[["param_classifier__max_depth", "mean_test_score", "std_test_score"]]
results = results.sort_values("mean_test_score", ascending=False)
print(results.to_string(index=False))

# Comments:
# max_depth=5 has the best mean AUC (0.9165) and a reasonably low std, so
# it's the clear winner - no tradeoff needed.
# Performance rises from depth=2 to depth=5 (too simple/underfit at shallow
# depths), then falls again as depth increases past 5 (overfitting).
# max_depth=None has both the lowest score and highest std (0.039) - a clearsign of overfitting.

# --- joblib ---
# Q1
os.makedirs('models', exist_ok=True)
joblib.dump(best_pipe, "models/warmup_model.pkl")
loaded_clf = joblib.load("models/warmup_model.pkl")
original_preds = best_pipe.predict(X_test)
loaded_preds   = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")
# Comments:
# if I saved only the LogisticRegression (without the scaler) and then
# called .predict() on unscaled X_test, it wouldn't throw an error - it
# would just silently give wrong predictions.
# the model learned its coefficients based on scaled features (mean=0, std=1), so feeding it raw
# unscaled data means each feature is now on a totally different range than
# what the model expects. the math still runs, but the results are
# meaningless.

# Q2
# --- Simulated prediction script ---
joblib.dump(best_pipe, "models/warmup_model.pkl")
loaded_clf = joblib.load("models/warmup_model.pkl")

new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

predictions = loaded_clf.predict(new_samples)
probabilities = loaded_clf.predict_proba(new_samples)[:, 1]

for i, (pred, prob) in enumerate(zip(predictions,probabilities )):
    label = "class 1" if pred == 1 else "class 0"
    print(f"Row {i+1}: predicted class = {label}, probability of class 1 = {prob:.3f}")
    
# Comments:
# Row 1: predicted class = 1, probability = 1.000
# Row 2: predicted class = 0, probability = 0.000
# Row 3: predicted class = 0, probability = 0.026
# I thought the all-zeros row would land close to 0.5, but it was 0.026.
# these are raw zeros, and the Pipeline scales them before the model sees
# them - so raw zeros don't stay "neutral" after scaling.
