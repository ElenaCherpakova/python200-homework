import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.inspection import DecisionBoundaryDisplay

warnings.filterwarnings("ignore", category=RuntimeWarning)

COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

# Task 1: Load and Explore

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
print(df.head())

features_to_plot = ['word_freq_free', 'char_freq_!', 'capital_run_length_total']

for p in features_to_plot:
    df.boxplot(column=p, by='spam_label')
    plt.title(f"Distribution of {p}")
    plt.suptitle("")  
    plt.xlabel("Class (0 = not spam, 1 = spam)")
    plt.ylabel(p)
    plt.savefig(f'outputs/boxplot_{p}.png')
    plt.show()
    
# Comments:
# Notice that all 3 features show the same pattern the box for spam sits higher than the box for ham
# Spam emails use word free more often and use more exclamation marks, and have longer runs of capital letters.
# All 3 have outliers in both classes.
# The differences are subtle rather than dramatic.
# Most emails don't contain any one specific word, so word-frequency features are mostly zeros.
# Scale varies a lot because features measure different things - frequencies are small percentages, 
# while capital-letter counts can reach into the thousands.
# This matters for models like KNN, which use distance: without scaling,
# big-range features like capital_run_length_total would dominate just
# because of their size, even if a smaller feature is more useful for
# telling spam from ham. So scaling is needed here too, same as with iris.

# Task 2: Prepare Your Data


X = df[COLUMN_NAMES[:-1]]
y = df['spam_label']

# --- Preprocessing --- 
# Q1

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42, stratify=y)

# Scale before PCA - features are on very different scales
# (word frequencies are tiny percentages, capital_run_length can be
# in the thousands), so without scaling, large-range features would dominate.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit scaler and PCA on training data only, to avoid leaking
# test-set information into preprocessing.
pca = PCA()
pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
perc_exp_vals = cumulative_variance * 100
n = np.where(perc_exp_vals >= 90)[0][0] + 1
print(n)

plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.savefig('outputs/spambase_pca_variance.png')
plt.show()

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]

# Task 3: A Classifier Comparison

knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)
preds= knn_raw.predict(X_test)
print("Accuracy for raw:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

knn_scalled = KNeighborsClassifier(n_neighbors=5)
knn_scalled.fit(X_train_scaled, y_train)
preds_scaled = knn_scalled.predict(X_test_scaled)
print("Accuracy for scaled:", accuracy_score(y_test, preds_scaled))
print(classification_report(y_test, preds_scaled))
print(X.describe())

knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
preds_pca = knn_pca.predict(X_test_pca)
print("Accuracy for PCA-reduced:", accuracy_score(y_test, preds_pca))
print(classification_report(y_test, preds_pca))


max_depth_list = [3, 5, 10, None]
for max_depth in max_depth_list:
    decision_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    decision_tree.fit(X_train, y_train)
    
    train_preds = decision_tree.predict(X_train)
    test_preds = decision_tree.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"max_depth={max_depth}: train accuracy={train_acc:.3f}, test accuracy={test_acc:.3f}")

# Comments: as max_depth increases, training accuracy keeps climbing
# (toward 1.0 at max_depth=None), but test accuracy improves only up to a
# point, then flattens or drops slightly. A growing gap between train and
# test accuracy is the signature of overfitting: the tree is memorizing
# training data rather than learning patterns that generalize.

final_depth = 10  
decision_tree_final = DecisionTreeClassifier(max_depth=final_depth, random_state=42)
decision_tree_final.fit(X_train, y_train)
final_preds = decision_tree_final.predict(X_test)
print(f"Final Decision Tree (max_depth={final_depth})")
print("Accuracy:", accuracy_score(y_test, final_preds))
print(classification_report(y_test, final_preds))


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print(f"Random Forest (n_estimators=100):")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))

log_reg_scaled = LogisticRegression(
            max_iter=1000, 
            solver="liblinear", 
            C=1.0
        )
log_reg_scaled.fit(X_train_scaled, y_train)
log_reg_scaled_preds = log_reg_scaled.predict(X_test_scaled)
print("Logistic Regression (scaled):")
print("Accuracy:", accuracy_score(y_test, log_reg_scaled_preds))
print(classification_report(y_test, log_reg_scaled_preds))


log_reg_pca = LogisticRegression(
            max_iter=1000, 
            solver="liblinear", 
            C=1.0
        )
log_reg_pca.fit(X_train_pca, y_train)
log_reg_pca_preds = log_reg_pca.predict(X_test_pca)
print("Logistic Regression (PCA-reduced):")
print("Accuracy:", accuracy_score(y_test, log_reg_pca_preds))
print(classification_report(y_test, log_reg_pca_preds))

#Comments:
# Random Forest wins clearly with 0.946, a solid margin above everything else.

cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['not spam', 'spam']
)
disp.plot()
plt.title("Random Forest Confusion Matrix (Best Model)")
plt.savefig('outputs/best_model_confusion_matrix.png')
plt.show()

# Comment: Random Forest performed best overall (accuracy 0.946), better
# than Decision Tree (0.909), Logistic Regression scaled (0.929) and PCA
# (0.919), and all 3 KNN versions (raw 0.799, scaled 0.908, PCA 0.907).
#
# For KNN and Logistic Regression, scaled (non-PCA) worked slightly better
# than PCA-reduced. This matches what we expected in Task 2 - PCA finds
# directions of max variance, not necessarily the directions most useful
# for telling spam from ham, so some signal gets lost in the compression.
#
# For a spam filter, I don't think accuracy alone is the right metric.
# False positive (real email marked as spam) is worse than false negative
# (spam that gets through), because a missed important email can have real
# consequences, while spam in the inbox is just annoying.
#
# My best model (Random Forest) makes more false negatives (32) than false
# positives (18) -- so it lets more spam through than it blocks real email.
# Given the cost tradeoff above, this is actually the safer direction to
# lean, even though it wasn't optimized for that directly.



rf_importances = pd.Series(rf.feature_importances_, index=X.columns)
dt_importances = pd.Series(decision_tree_final.feature_importances_, index=X.columns)
print("Top 10 features - Random Forest:")
print(rf_importances.sort_values(ascending=False).head(10))

print("Top 10 features - Decision Tree:")
print(dt_importances.sort_values(ascending=False).head(10))

top_10_rf = rf_importances.sort_values(ascending=False).head(10)
top_10_rf.plot(kind='bar')
plt.title("Top 10 Feature Importances - Random Forest")
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.tight_layout()  # prevents feature names from getting cut off
plt.savefig('outputs/feature_importances.png')
plt.show()
# Comment: The two models agree on 6 of the top 10 features (char_freq_$,
# char_freq_!, word_freq_remove, word_freq_free, capital_run_length_total,
# word_freq_hp), with Decision Tree concentrating importance on fewer
# features (single best-split logic) versus Random Forest spreading it
# more evenly (averaging over many trees), and the top features (dollar
# signs, exclamation marks, "remove", "free", all-caps) match intuition
# about what makes an email look spammy.



# Task 4: Cross-Validation

models_to_test = [
    ("KNN (raw)", KNeighborsClassifier(n_neighbors=5), X_train),
    ("KNN (scaled)", KNeighborsClassifier(n_neighbors=5), X_train_scaled),
    ("KNN (PCA)", KNeighborsClassifier(n_neighbors=5), X_train_pca),
    ("Decision Tree (depth=10)", DecisionTreeClassifier(max_depth=10, random_state=42), X_train),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42), X_train),
    ("Logistic Regression (scaled)", LogisticRegression(max_iter=1000, solver="liblinear", C=1.0), X_train_scaled),
    ("Logistic Regression (PCA)", LogisticRegression(max_iter=1000, solver="liblinear", C=1.0), X_train_pca),
]


for name, model, X_data in models_to_test:
    scores = cross_val_score(model, X_data, y_train, cv=5)
    print(f"{name}: mean={scores.mean():.3f}, std={scores.std():.3f}")
    
# Comment: Random Forest is the most accurate (mean=0.954), same as in
# Task 3. Logistic Regression (PCA) is the most stable (std=0.003) -
# lowest variance across folds. Random Forest also has lower variance
# than Decision Tree (0.013 vs 0.019), matching what was expected.
# The overall ranking is close to what I saw with the single train/test
# split, which makes me more confident those Task 3 results weren't just
# luck from one split.

# Task 5: Building a Prediction Pipeline

tree_pipeline = Pipeline([
    ("classifier",  RandomForestClassifier(n_estimators=100, random_state=42))])

non_tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')),
])

tree_pipeline.fit(X_train, y_train)
tree_preds = tree_pipeline.predict(X_test)
print("Tree Pipeline (Random Forest):")
print(classification_report(y_test, tree_preds))

non_tree_pipeline.fit(X_train, y_train)
non_tree_preds = non_tree_pipeline.predict(X_test)
print("Non-Tree Pipeline (Logistic Regression, scaled):")
print(classification_report(y_test, non_tree_preds))

# Comment: The two pipelines have different structure -- tree pipeline
# skips the scaler because trees split on thresholds and don't care about
# feature scale, while the non-tree pipeline needs a scaler because
# Logistic Regression is sensitive to feature magnitude.