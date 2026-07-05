import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns



data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}

df = pd.DataFrame(data)
# --- Pandas
# --- Pandas Q1 ---
# --- first three rows ---
print(f"First three rows: {df.head(3)}")
# ---  the shape ---
print(f"Shape: {df.shape}")
# --- data types of each column
print(f"Data types of each column: {df.dtypes}")

# --- Pandas Q2 ---
filtered_df = df.loc[ (df['passed']) & (df['grade'] > 80), ['name', 'grade'] ]
print(f"Students who passed and have grade > 80:\n {filtered_df}")

# --- Pandas Q3 ---
df2 = df.copy()
df2["grade_curved"] = df2['grade'] + 5
print(f"DataFrame with curved grades:\n {df2}")

# --- Pandas Q4 ---
df2['name_upper'] = df2['name'].str.upper()
print(f"DataFrame with uppercase names:\n {df2}")

# --- Pandas Q5 ---
average_grades = df2.groupby('city')['grade'].mean().reset_index()
print(f"Average grades by city:\n {average_grades}")

# --- Pandas Q6 ---
df2['city'] = df2['city'].replace('Austin', 'Houston')
print(f"DataFrame with updated city names:\n {df2[['city', 'name']]}")

# --- Pandas Q7 ---
top_students = df2.sort_values('grade', ascending=False).head(3)
print(f"Top 3 students by grade:\n {top_students}")

# --- NumPy Review ---
# --- NumPy Q1 ---
arr = np.array([10, 20, 30, 40, 50])
print(f"shape of the array {arr.shape}")
print(f"array type: {arr.dtype}")
print(f"number of dimensions (ndim): {arr.ndim}")

# --- NumPy Q2 ---
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(f"Shape of the array {arr.shape}")
print(f"Size of the array: {arr.size}")

# --- NumPy Q3 ---
arr_reshaped = arr[0:2, 0:2]
print(f"Reshaped array: {arr_reshaped}")

# --- NumPy Q4 ---
arr_of_zeros = np.zeros((3, 4), dtype=int)
print(f"Array of zeros:\n {arr_of_zeros}")
arr_of_ones = np.ones((2, 5), dtype=int)
#another way to create an array of ones is to use np.full
# arr_of_ones = np.full((2, 5), 1, dtype=int)
print(f"Array of ones:\n {arr_of_ones}")

# --- NumPy Q5 ---
array_range = np.arange(0, 50, 5)
print(f"Array with range: {array_range}")
print(f"Shape: {array_range.shape}")
print(f"Mean: {array_range.mean()}")
print(f"Sum: {array_range.sum()}")
print(f"Standard Deviation: {array_range.std()}")

# --- NumPy Q6 ---
arr_200 = np.random.normal(0, 1, 200)
print(f"Mean of 200 random numbers: {arr_200.mean()}")
print(f"Standard Deviation of 200 random numbers: {arr_200.std()}")


# --- Matplotlib Review ---
# --- Matplotlib Q1 ---
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.plot(x, y)
plt.title('Squares')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# --- Matplotlib Q2 ---
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
plt.bar(subjects, scores)
plt.title('Subject Scores')
plt.xlabel('Subjects')
plt.ylabel('Scores')
plt.show()

# --- Matplotlib Q3 ---
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color='blue', label='Group A')
plt.scatter(x2, y2, color='red', label='Group B')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# --- Matplotlib Q4 ---
fig, axes = plt.subplots(1, 2)
axes[0].plot(x, y)
axes[0].set_title('Squares')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

axes[1].bar(subjects, scores)
axes[1].set_title('Subject Scores')
axes[1].set_xlabel('Subjects')
axes[1].set_ylabel('Scores')
plt.tight_layout()
plt.show()


# --- Descriptive Statistics Review ---
# --- Descriptive Statistics Q1 ---
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
mean_data = np.mean(data)
print(f"Mean: {mean_data}")
median_value = np.median(data)
print(f"Median: {median_value}")
variance = np.var(data)
print(f"Variance: {variance}")
standard_deviation = np.std(data)
print(f"Standard Deviation: {standard_deviation}")

# --- Descriptive Statistics Q2 ---
data_q2 = np.random.normal(65, 10, 500)
plt.hist(data_q2, bins=20)
plt.title('Distribution of Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()

# --- Descriptive Statistics Q3 ---
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
data_to_compare = [group_a, group_b]
fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot(data_to_compare, tick_labels=['Group A', 'Group B'])
ax.set_title('Score Comparison')
ax.set_ylabel('Values')
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# --- Descriptive Statistics Q4 ---
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
data_to_compare = [normal_data, skewed_data]
print(f"Normal data mean: {np.mean(normal_data)}, std: {np.std(normal_data)}")
print(f"Exponential data mean: {np.mean(skewed_data)}, std: {np.std(skewed_data)}")
fig, ax =  plt.subplots(figsize=(8, 6))
ax.boxplot(data_to_compare, tick_labels=['Normal', 'Exponential'])
ax.set_title('Distribution Comparison')
ax.set_ylabel('Values')
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# --- Distribution comparison ---
# Normal data is symmetric (not skewed) -> mean is a good measure of center.
# Exponential data is right-skewed (long tail of high outliers, confirmed by
# boxplot) -> median is a better measure of center since mean gets pulled up
# by the outliers.

# --- Descriptive Statistics Q5 ---
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]
data1_mean = np.mean(data1)
data2_mean = np.mean(data2)
data1_median = np.median(data1)
data2_median = np.median(data2)
mode_data1 = pd.Series(data1).mode()[0]
mode_data2 = pd.Series(data2).mode()[0]

print(f"Data - 1: Mean: {data1_mean}, Median: {data1_median}, Mode: {mode_data1}")
print(f"Data - 2: Mean: {data2_mean}, Median: {data2_median}, Mode: {mode_data2}")

# Data2's mean (40.0) is much higher than data1's (13.6) because of the 
# outlier 150 - mean is sensitive to extreme values.
# Median (12.0) stays the same in both, since it only depends on the
# middle value's position, not how big the numbers are.


# --- Hypothesis ---
# --- Hypothesis Q1 ---
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

result = stats.ttest_ind(group_a, group_b) # independent samples, we use ttest_ind
print(f"Q1: t-statistic: {result.statistic}")
print(f"Q1: p-value: {result.pvalue}")

# --- Hypothesis Q2 ---
if result.pvalue <= 0.05:
    print('You reject the null hypothesis. The difference is statistically significant.')
else:
    print('You fail to reject the null hypothesis. There is not enough evidence to prove a significant difference.')
    
# --- Hypothesis Q3 ---
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]
result = stats.ttest_rel(before, after) # related samples, we use ttest_rel
print(f"t-statistic: {result.statistic}")
print(f"p-value: {result.pvalue}")

# --- Hypothesis Q4 ---
scores = [72, 68, 75, 70, 69, 74, 71, 73]
result = stats.ttest_1samp(scores, 70)
print(f"t-statistic: {result.statistic}")
print(f"p-value: {result.pvalue}")

# --- Hypothesis Q5 ---
result = stats.ttest_ind(group_a, group_b, alternative="less")
print(f"p-value: {result.pvalue}")

# --- Hypothesis Q6 ---
print("Group A scored lower than Group B on average, and this difference is extremely unlikely to be due to random chance (p ≈ 0.0000015), suggesting a real effect.")

# --- Correlation Review ---
# --- Correlation Q1 ---
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
corr_matrix = np.corrcoef(x, y)
correlation = corr_matrix[0, 1]
print(f"Correlation matrix: {corr_matrix}")
print(f"Coefficient: {correlation}")


# The correlation to be 1 (or very close to it) because y = 2x
# every increase in x produces a perfectly predictable, proportional increase
# in y, with no randomness or scatter in the data.

# --- Correlation Q2 ---
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]
coef, p = pearsonr(x, y)

print(f"Correlation coefficient: {coef}")
print(f"Correlation p-value: {p}")

# --- Correlation Q3 ---
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
correlation_matrix_q3 = df.corr()
print(f"Correlation matrix:\n{correlation_matrix_q3}")

# --- Correlation Q4 ---
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
plt.scatter(x, y, color='blue')
plt.title('Negative Correlation')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.show()

# --- Correlation Q5 ---
sns.heatmap(correlation_matrix_q3, annot=True)
plt.title('Correlation Heatmap')
plt.show()

# --- Pipelines ---
# --- Pipelines Q1 ---
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])


def create_series(arr):
    series = pd.Series(arr, name="values")
    return series

def clean_data(series):
    cleaned_series = series.dropna()
    return cleaned_series

def summarize_data(series):
    return {
    "mean": series.mean(),
    "median": series.median(),
    "std": series.std(),
    "mode": series.mode()[0]
    }
    
def data_pipeline(arr):
    series = create_series(arr)
    cleaned_data = clean_data(series)
    result = summarize_data(cleaned_data)
    return result
    
result = data_pipeline(arr)
for key, value in result.items():
    print(f"{key}: {value}")
    

