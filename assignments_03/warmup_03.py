import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing --- 
# Q1

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train:", X_train.shape)
print(f"X_test:", X_test.shape)
print(f"y_train:", y_train.shape)
print(f"y_test:", y_test.shape)

# Q2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mean_x_train_scaled = X_train_scaled.mean(axis=0)
mean_y_test_scaled =  X_test_scaled.mean(axis=0)
print(mean_x_train_scaled)
print(mean_y_test_scaled)

#Comments: 
# We fit the scaler on X_train in order to prevent data leakage and avoid test data 
# to influence on mean and standard deviation of each feature.

# KNN
# Q1
knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)
preds= knn_raw.predict(X_test)
print("Accuracy for raw:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Q2
knn_scalled = KNeighborsClassifier(n_neighbors=5)
knn_scalled.fit(X_train_scaled, y_train)
preds_scaled = knn_scalled.predict(X_test_scaled)
print("Accuracy for scaled:", accuracy_score(y_test, preds_scaled))
print(classification_report(y_test, preds_scaled))
print(X.describe())


#Comment: Raw version scored 1 and scaled one is 0.93 - 2 classified points out of 30
# Not a dramatic failure. This does not mean scaling hurts KNN 
# standardizing them didn't fix any real scale imbalance.
# On such a small test set with 30 samples a 2-point difference is likely
# noise rather than evidence scaling is harmful. In general, KNN still
# benefits from scaling when features have very different units/ranges.

# Q3
cv_scores = cross_val_score(knn_raw, X_train, y_train, cv=5)
print(cv_scores) # accuracy on each fold
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std:  {cv_scores.std():.3f}")

# Comments: 
# Mean closer to 0.97 - excellent, but a more honest picture than a single 1.0.
# The average score is better than any single split because 
# cross-validation gives a more reliable/trustworthy estimate
# it averages over 5 different train/test arrangements, rather than depending on the luck of a single split
# The low std 0.033 shows the model accuracy is consistent across folds not wildly different
# form one fold to the next - so the strong mean score isn't driven by one lucky fold.

# Q4
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}  std={scores.std():.3f}")
    
# Comments: 
# Best k=7 which is tied for highest mean at 0.975, but lower std than k=5 more consistent across folds


# Classifier Evaluation
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)
disp.plot()
plt.title("KNN Confusion Matrix (Iris)")
plt.savefig('outputs/knn_confusion_matrix.png')
plt.show()
# Comments: 
# This is a perfect confustion matrix. 
# There is zero confusion between any classes - the model got all 30 test samples right.

# The sklearn API: Decision Trees

decision_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
decision_tree.fit(X_train, y_train)
preds_descition_tree = decision_tree.predict(X_test)
print("Accuracy for decision_tree:", accuracy_score(y_test, preds_descition_tree))
print(classification_report(y_test, preds_descition_tree))

# Comments:
# 1. Decision Tree vs KNN:
# Decision Tree accuracy (0.967) is close to but slightly below KNN's CV mean (0.975) 
# a difference of about 1 misclassified point out of 30,
# likely within noise given the small test set, but on this data KNN still edges it out slightly.

# 2. Scaling would not affect a Decision Tree's result, because trees only care whether a value is bigger 
# or smaller than some threshold, and scaling doesn't change the order of the data — 
# it just shifts and stretches the numbers, so points that were above or below a given threshold
# stay above or below it

# Logistic Regression and Regularization
# Q1


C = [0.01, 1.0, 100]
for C in [0.01, 1.0, 100]: 
    log_reg = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000, 
            solver="liblinear", 
            C=C
            )
        )
    log_reg.fit(X_train_scaled, y_train)
    coef_sum = np.abs(np.vstack([est.coef_ for est in log_reg.estimators_])).sum()
    print(f"C={C}: total coef magnitude = {coef_sum:.3f}")

# Comment: As C increases, the total coefficient magnitude increased
# (1.965 -> 12.485 -> 37.890). A small C (like 0.01) applies strict
# regularization, pushing all coefficients toward zero. A large C (like 100)
# applies weak regularization, letting coefficients grow freely.
# This means small C risks underfitting (the model is too constrained to
# learn real patterns), while large C risks overfitting (the model fits
# the training data too closely, including noise).

# PCA
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# Q1
print(f"X_digits shape: {X_digits.shape}")
print(f"Images shape: {images.shape}")
for digit in range(10):
    for i in range(len(y_digits)):
        if(y_digits[i] == digit):
            plt.subplot(1, 10, digit + 1)
            plt.imshow(images[i], cmap='gray_r')
            plt.title(digit)
            break
plt.savefig('outputs/sample_digits.png')
plt.show()

# Q2
pca = PCA()
scores = pca.fit_transform(X_digits)
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10) 
plt.colorbar(scatter, label='Digit')
plt.savefig('outputs/pca_2d_projection.png')
plt.show()

# Comment: images do partially cluster together in this 2D space. Some digits form clear, separate
# clusters like 0, 4, and 6  because they look very different from
# other digits. But others overlap a lot in the middle, like 5, 7, and 8,
# because they can look similar in pixel shape. This makes sense since
# PC1 and PC2 are only 2 out of 64 total dimensions, so they can't fully
# separate every digit.

# Q3
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.savefig('outputs/pca_variance_explained.png')
plt.show()

perc_exp_vals = cumulative_variance * 100
n_components_80 = np.where(perc_exp_vals >= 80)[0][0] + 1
print(f"Components needed for 80%: {n_components_80}")

# Comments:
# Appx 13 out of the original 64 pixel-features we need to explain 80% of the variance

# Q4
values = [2, 5, 15, 40]
n_digits = 5
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

fig, axes = plt.subplots(nrows=len(values) + 1, ncols=n_digits, figsize=(10, 12))
# Row 0: originals
for col in range(n_digits):
    axes[0, col].imshow(images[col], cmap='gray_r')
    axes[0, col].set_title(f"Digit {y_digits[col]}")
    axes[0, col].axis("off")
axes[0, 0].set_ylabel("Original", rotation=0, labelpad=40)

# Remaining rows: reconstructions for each n in values
for row, n in enumerate(values, start=1):
    for col in range(n_digits):
        recon = reconstruct_digit(col, scores, pca, n)
        axes[row, col].imshow(recon, cmap='gray_r')
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(f"n={n}", rotation=0, labelpad=30)
    
plt.savefig('outputs/pca_reconstructions.png')
plt.show()

# Comments: 
# Digits look clearly recognizable starting around n=15. That's
# close to the 13 components we found earlier for 80% variance so the
# point where the variance curve flattens out roughly matches the point
# where the images start looking clear. After that (like n=40), more
# components just add small extra detail, not big changes.