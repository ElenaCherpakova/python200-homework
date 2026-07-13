# --- scikit-learn API ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# --- Q1 ----
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

experience_4 = [[4]]
experience_8 = [[8]]
# Supervised - seeing questing and answers simulteniously .fit(X, y) <- 2 arguments
model = LinearRegression()
model.fit(years, salary)

predict_4 = model.predict(experience_4)
predict_8 = model.predict(experience_8)

print(f"Slope {model.coef_[0]}") # one number for each column
print(f"Intercept {model.intercept_}")
print(f"Prediction for 4 years: {predict_4[0]}") #one number for each row
print(f"Prediction for 8 years: {predict_8[0]}")


# --- Q2 ----
x = np.array([10, 20, 30, 40, 50])
print(f"Shape: {x.shape}")
print(f"Reshape in 2D: {x.reshape(-1, 1)}")

# sklearn needs X to be 2D because a 1D array is ambiguous —
# it doesn't say whether the numbers are 5 samples with 1 feature each,
# or 1 sample with 5 features. The 2D shape (rows=samples, cols=features)
# removes that ambiguity.

# --- Q3 ----

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

#Unsupervised - seeing 1 array and there is no right answer, nobody teaches the model. .fit(X) <- 1 argument
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_clusters)
centroids = kmeans.cluster_centers_
print(f"Cluster center: {centroids}")
labels = kmeans.labels_
cluster_counts = np.bincount(labels)
for cluster_id, count in enumerate(cluster_counts):
    print(f"Cluster {cluster_id}: {count} data points")


plt.figure(figsize=(8, 6))
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=cluster_labels, s=50, cmap='viridis', alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='X', label='Centroids')
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.legend()
plt.savefig('outputs/kmeans_clusters.png')
plt.close()


# --- Liner Regression ---
np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# --- Q1 ----
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel('Age')
plt.ylabel('Cost')
plt.savefig('outputs/cost_vs_age.png')
plt.close()

# Red and blue dots form two clearly separated bands
# Cost still upward with age within each color group
# Since smokers sit noticeably higher on the cost axis than non-smokers 
# at the same age, this suggests that smoker status has a strong effect on cost

# --- Q2 ----

X_train, X_test, y_train, y_test = train_test_split(
    age.reshape(-1, 1), cost, test_size=0.2, random_state=42
)
print(f"X_train_shape: {X_train.shape}")
print(f"X_test_shape: {X_test.shape}")
print(f"Y_train_shape: {y_train.shape}")
print(f"Y_test_shape: {y_test.shape}")

# --- Q3 ----

model = LinearRegression()
model.fit(X_train, y_train)
print(f"Slope {model.coef_[0]}") 
print(f"Intercept {model.intercept_}")
y_pred =  model.predict(X_test)
np.sqrt(np.mean((y_pred - y_test) ** 2))
print(f"Test R2 (age only): {model.score(X_test, y_test)}")


# For each additional year of age, cost tends to grow by about $196
# this only accounts for age, model doesnt know about smoking which has a big effect on smoke.

# --- Q4 ----
X_full = np.column_stack([age, smoker])
model_full = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(
    X_full, cost, test_size=0.2, random_state=42
)

model_full.fit(X_train, y_train)
score_full = model_full.score(X_test, y_test)
print(f"Test R² (age + smoker): {score_full}")
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])
# Smoker as a feature raised the test R² noticeable higher-> bigger share of the variation cost.
# 14,538 higher medical cost compared to non-smoker of the same age.
# Q3 (age only) R² was 0.07, meaning age alone explained only about 7% of cost variation.
# Q4 (age + smoker) R² was 0.77, explaining about 77%. 
# This is a huge improvement, confirming that cost depends heavily on smoking status,
# not just age.

y_pred_full = model_full.predict(X_test)
plt.scatter(y_pred_full, y_test)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="black")

plt.title("Predicted vs Actual")
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.savefig('outputs/predicted_vs_actual.png')
# Above the diagonal: model underestimated the cost.
# Below the diagonal: model overestimated the cost.