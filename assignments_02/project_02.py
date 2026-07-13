import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import mean_squared_error, r2_score


# Task 1: Load and Explore

file = 'student_performance_math.csv'
df = pd.read_csv(file, delimiter=";")
print("Shape:", df.shape)
print("\nfirst 5 and data types:\n", df.head(5))
print("\nData types:\n", df.dtypes)

plt.hist(df['G3'], bins=21, range=(0, 20), edgecolor='black')
plt.title("Distribution of Final Math Grades")
plt.xlabel('Final grade G3')
plt.ylabel('Number of students')
plt.savefig('outputs/g3_distribution.png')

print("Shape before:", df.shape)
df_nonzero=df[df['G3'] != 0].copy()
print("Shape after", df_nonzero.shape)
# Output: 
# Shape before: (395, 18)
# Shape after (357, 18)
# G3 = 0 is probably because the student withdrew from the course or was
# absent on exam day, not because they actually received a score of 0.
# If we leave these rows, the model will think that students with average
# study habits, attendance, etc. can still "fail" with a score of 0, when
# in reality the cause is missing data, not a genuinely low score.

# Task 2: Preprocess the Data
yes_no_cols = ['schoolsup', 'internet', 'higher', 'activities']
for col in yes_no_cols:
    df_nonzero[col] = df_nonzero[col].map({'yes': 1, 'no': 0});

df_nonzero['sex'] = df_nonzero['sex'].replace({'F': 0, 'M': 1});
print(df['sex'], df_nonzero['sex'] );
coef_original, p_value_original = pearsonr(df['absences'], df['G3'])
coef_filtered, p_value_filtered = pearsonr(df_nonzero['absences'], df_nonzero['G3'])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(df['absences'], df['G3'], alpha=0.5)
axes[0].set_title("Original (includes G3=0)")
axes[0].set_xlabel("Absences")
axes[0].set_ylabel("G3")

axes[1].scatter(df_nonzero['absences'], df_nonzero['G3'], alpha=0.5)
axes[1].set_title("Filtered (G3=0 removed)")
axes[1].set_xlabel("Absences")
axes[1].set_ylabel("G3")

plt.tight_layout()
plt.show()
print(f"coef_original: {coef_original}")
print(f"p_value_original: {p_value_original}")

print(f"coef_filtered: {coef_filtered}")
print(f"p_value_filtered: {p_value_filtered}")
# Output:
# coef_original: 0.034247316150069325
# p_value_original: 0.497331795543527
# coef_filtered: -0.21312853214380884
# p_value_filtered: 4.916538246846149e-05
#Comments:
# For raw data, p > 0.05 (p = 0.497), so the correlation between absences
# and G3 is NOT statistically significant. For filtered data, p < 0.05
# (p = 0.00005), so the correlation is statistically significant.
# G3=0 doesn't reliably indicate real academic performance, because there
# are several possible reasons behind it - the student may have withdrawn,
# dropped the course before reaching the final exam, or been absent on
# exam day. Since we can't tell which reason applies, treating G3=0 as a
# genuine "zero score" is misleading.

# Task 3: Exploratory Data Analysis

numeric_cols = df_nonzero.select_dtypes(include='number').columns
numeric_cols = numeric_cols.drop('G3')

correlation = {}
for col in numeric_cols:
    coef, p_value = pearsonr(df_nonzero[col], df_nonzero['G3'])
    correlation[col] = {'coef': coef, 'p_value': p_value}
    
corr_df = pd.DataFrame(correlation).T
corr_df = corr_df.sort_values('coef')
print(corr_df)

# --- Plot 1: failures vs G3 (boxplot) ---
# 'failures' has the strongest correlation with G3 among non-grade features
# (coef = -0.294, p = 1.5), so it's worth a closer look.
df_nonzero.boxplot(column='G3', by='failures', figsize=(6, 5))
plt.title("Final Grade (G3) by Number of Past Failures")
plt.suptitle('')
plt.xlabel("Number of Past Class Failures")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig('outputs/failures_vs_g3.png')
plt.show()

# Comments:
# Comment: Median G3 drops steadily as failures increase: 11.5 for 0
# failures, 10 for 1, 8.5 for 2, and 8 for 3. The spread also narrows -
# students with 0 failures range widely (5-20), while students with 2-3
# failures are tightly clustered in the 7-10 range with a much lower
# ceiling. A few outliers at failures=2 (scores of 13, 15) show some
# students still do well despite past failures, but the overall trend
# is a clear step down at each failure level - consistent with 'failures'
# being the strongest non-grade predictor in corr_df.


# --- Plot 2: schoolsup vs G3 (boxplot) ---
# 'schoolsup' (extra educational support) shows the second-strongest
# correlation (coef = -0.238, p = 5.3e-06) - negative,
# worth investigating why.
df_nonzero.boxplot(column='G3', by='schoolsup', figsize=(6, 5))
plt.title("Final Grade (G3) by School Support")
plt.suptitle('')
plt.xlabel("Receives Extra School Support (0=No, 1=Yes)")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig('outputs/schoolsup_vs_g3.png')
plt.show()
# Comment:
# Students without extra school support (0) have a higher
# median G3 (11) and a wider top range (up to 20) than students who
# receive support (1), whose median sits around 9-10 and rarely exceeds
# 15 (one outlier at 17). This is counterintuitive, support is
# associated with lower grades. The likely explanation is reverse
# causation: schoolsup is probably assigned to students who were already
# struggling, so it's a response to low performance rather than a cause
# of it. This is a good example of correlation not implying causation.

# Task 4: Baseline Model

model = LinearRegression()
#Split first using failures as the single feature
X_train, X_test, y_train, y_test = train_test_split(
    df_nonzero[['failures']], df_nonzero['G3'], test_size=0.2, random_state=42
)
# Fit only on training data
model.fit(X_train, y_train)
# Predict on test data
y_pred = model.predict(X_test)
# Metrics 
slope = model.coef_[0]
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Slope: {slope}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")
# Output:
# Slope: -1.4275148751598754 
# RMSE: 2.9617372470468797 
# R²: 0.08949272357478744 
# Yes, past failures do genuinely affect the grade 
# (it's not random chance — the p-value is small). But the effect is moderate, not huge (coef = -0.29).
# And if we build a model based on this one factor alone, it only explains
# about 9% of the differences between students — the rest depends on other things.

# Task 5: Build the Full Model
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
X = df_nonzero[feature_cols].values
y = df_nonzero["G3"].values
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2_test = r2_score(y_test, y_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
y_train_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)

print(f"Train R2: {r2_train}")
print(f"Test R2:  {r2_test}")
print(f"Test RMSE: {rmse_test}")

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")
    
# Comments:

# failures    : -1.145
# Medu        : +0.083
# Fedu        : +0.186
# studytime   : +0.448
# higher      : +0.610
# schoolsup   : -2.062 - biggest effect, but negative. Likely reversed causation support is given to student
# who are already struggle, not the cause of low grades. 
# internet    : +0.834 - bigger boost that expected can be possibility of a proxy for household resources rather than a direct cause.
# sex         : +0.453
# freetime    : -0.042
# activities  : -0.009
# traveltime  : -0.112

# Test R² went from 0.089 (failures alone) to 0.154 with 11 features which is
# better, but still modest. These 11 features only explain about 15%
# of why grades differ between students likely because we excluded
# G1 and G2, which had by far the strongest correlation with G3
# Train 0.175 vs Test 0.154 small gap, so no real overfitting.
# The model generalizes about as well on new data as on training data,
# just weakly overall.

# --- Production features ---
# KEEP: failures, schoolsup, higher, internet, studytime - biggest
# effects and match EDA or have a plausible explanation.
# DROP: activities, freetime - near-zero coefficients, not significant in EDA either.

# Task 6: Evaluate and Summarize

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y_test, alpha=0.5)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Predicted = Actual')
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.legend()
plt.tight_layout()
plt.savefig('outputs/predicted_vs_actual.png')
plt.show()

# Comments: The model struggles most at both ends of the
# grade scale, not in the middle. For high actual grades (15-19), points
# sit above the diagonal - the model guesses too low. For low actual
# grades (5-7), points sit below the diagonal - the model guesses too
# high. In the middle range (9-13), predictions are more accurate.
# A point above the diagonal means the model under-predicted (real
# grade was higher). A point below the diagonal means the model
# over-predicted (real grade was lower).

# Summary:

# Dataset size: after removing G3=0 rows, we had 357 students total.
# The test set 20% had about 72 students; the rest 285 students were used
# for training.

# Model performance: RMSE = 2.86, R2 = 0.154. On a 0-20 scale, this
# means our predictions are typically off by about 3 points - so if
# the model predicts a 12, the real grade is probably somewhere around
# 9 to 15. The model explains only about 15% of why grades differ
# between students - it's a weak predictor, not a strong one.

# Largest positive coefficient: internet (+0.834) - students with
# internet at home tend to score higher.
# Largest negative coefficient: schoolsup (-2.062) - students who get
# extra school support tend to score lower.

# Most surprising result: schoolsup having a negative effect. Extra
# support should logically help students, not hurt them. The likely
# explanation is that support is given to students who are already
# struggling, so it looks like it's causing lower grades when really
# it's the other way around - low grades caused the support to be given.

# Neglected Feature: The Power of G1

feature_cols_w_g1 = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime", "G1"]

X = df_nonzero[feature_cols_w_g1].values
y = df_nonzero["G3"].values
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2_test = r2_score(y_test, y_pred)
print(f"Test R2:  {r2_test}")
for name, coef in zip(feature_cols_w_g1, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")
    
#Comments:
# Does high R² mean G1 causes G3? No - G1 and G3 are just two
# measurements of the same thing (how well a student is doing overall).
# A good G1 doesn't cause a good G3, they both come from the same
# student ability/habits.

# Is this useful for catching struggling students early? Not really -
# G1 only exists after the first grading period is already over. By then it might be too late to help.

# What could educators use for real early intervention? Features
# available before any grades exist - failures, schoolsup, internet,
# parental education, study habits. Our earlier model without G1 only
# had R²=0.154 - much weaker, but it's the realistic option if you
# want to catch struggling students early, since G1 just isn't
# available yet at that point.