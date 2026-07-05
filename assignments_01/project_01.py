import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from prefect import task, flow
from prefect.logging import get_run_logger
import os


data_dir = "happiness_project"
years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
file_name = 'world_happiness'
output_path = 'outputs/merged_happiness.csv'
visual_output_path = 'outputs'


# Task 1: Load Multiple Years of Data
@task(task_run_name="Load Data")
def load_data(arr):
    res = []
    logger = get_run_logger()
    logger.info('Loading data...')
    for year in arr:
        file_path = os.path.join(data_dir, f"{file_name}_{year}.csv")
        try:
            df = pd.read_csv(file_path, sep=';', decimal=',', encoding='utf-8')
            # create a new column 'year' and assign the current year to it
            df['year'] = year
            res.append(df)
            logger.info(f"Data for {year} loaded successfully.")
        except FileNotFoundError:
            logger.error(f"File {file_path} not found.")
        except Exception as e:
            logger.error(f"An error occurred while loading data for {year}: {e}")
    return res
        

@task(task_run_name="Combine Data")
def combine_data(res):
    logger =  get_run_logger()
    logger.info('Combining data...')
    combined_df = pd.concat(res, ignore_index=True)
    logger.info('Data combined successfully.')
    return combined_df


@task(task_run_name='Clean Columns')
def clean_columns(df):
    logger = get_run_logger()
    logger.info('Cleaning columns names...')
    df.columns= (df.columns.str.strip().str.lower().str.replace(' ', '_'))
    if 'happiness_score' not in df.columns:
        df['happiness_score'] = df['ladder_score']
    if 'ladder_score' in df.columns and 'happiness_score' in df.columns:
        df['happiness_score'] = df['happiness_score'].fillna(df['ladder_score'])
        df = df.drop(columns=['ladder_score'])
        logger.info('Combine happiness_score and ladder_score into one columns')
    logger.info('Columns names cleaned successfully.')
    return df
@task (task_run_name="Save Data", retries=3, retry_delay_seconds=2)
def save_data(df, output_file):
        logger = get_run_logger()
        logger.info(f'Saving combined data to {output_file}...')
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            logger.info(f'Data saved successfully to {output_file}.')
        except Exception as e:
            logger.error(f"An error occurred while saving data to {output_file}: {e}") 
            raise
        
# Task 2: Descriptive Statistics
@task(task_run_name="Descriptive Statistics")  
def compute_stats(df):
    logger = get_run_logger()
    logger.info('Computing descriptive statistics...')
    
    overall_mean = df['happiness_score'].mean()
    overall_median = df['happiness_score'].median()
    overall_std = df['happiness_score'].std()
    
    logger.info(f"Overall happiness_score mean: {overall_mean:.3f}")
    logger.info(f"Overall happiness_score median: {overall_median:.3f}")
    logger.info(f"Overall happiness_score std: {overall_std:.3f}")
    
    by_year = df.groupby('year')['happiness_score'].mean()
    logger.info(f"Mean happiness_score by year:\n{by_year}")
    
    by_region = df.groupby('regional_indicator')['happiness_score'].mean().sort_values(ascending=False)
    logger.info(f"Mean happiness_score by region:\n{by_region}")
    logger.info('Descriptive statistics computed successfully.')
    
    return {
        'mean': overall_mean,
        'median': overall_median,
        'std': overall_std,
        'by_year': by_year,
        'by_region': by_region
    }

# Task 3: Visual Exploration
@task(task_run_name="Create Visualizations")
def create_visualization(df, output_dir=visual_output_path):
    logger = get_run_logger()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Histogram
    plt.figure(figsize=(8,6))
    plt.hist(df['happiness_score'])
    plt.xlabel('Happiness Score')
    plt.ylabel("Frequency")
    plt.title("Distribution of Happiness Scores (All Years)")
    plt.savefig(os.path.join(output_dir, 'happiness_histogram.png'))
    plt.close()
    logger.info('Saved happiness_histogram.png')
    
    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="year", y="happiness_score", data=df)
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.title("Happiness Score Distribution by Year")
    plt.savefig(os.path.join(output_dir, "happiness_by_year.png"))
    plt.close()
    logger.info("Saved happiness_by_year.png")

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df["gdp_per_capita"], df["happiness_score"], alpha=0.5)
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")
    plt.title("GDP per Capita vs Happiness Score")
    plt.savefig(os.path.join(output_dir, "gdp_vs_happiness.png"))
    plt.close()
    logger.info("Saved gdp_vs_happiness.png")
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()
    logger.info("Saved correlation_heatmap.png")
    
# Task 4: Hypothesis Testing


@task(task_run_name="Hypothesis Testing")
def hypothesis_test(df):
    logger = get_run_logger()
    
    scores_2019 = df[df['year'] == 2019]["happiness_score"]
    scores_2020 = df[df['year'] == 2020]["happiness_score"]
    

    result_year = stats.ttest_ind(scores_2019, scores_2020)
    
    mean_2019 = scores_2019.mean()
    mean_2020 = scores_2020.mean()
    
    logger.info(f"2019 mean happiness: {mean_2019:.3f}")
    logger.info(f"2020 mean happiness: {mean_2020:.3f}")
    logger.info(f"t-statistic: {result_year.statistic:.3f}")
    logger.info(f"p-value: {result_year.pvalue:.4f}")

    alpha = 0.05
    if result_year.pvalue < alpha:
        direction = "higher" if mean_2020 > mean_2019 else "lower"
        interpretation_year = (
            f"p-value = {result_year.pvalue:.4f}, which is below {alpha}, (we reject the null hypothesis), so this result is "
            f"statistically significant. Happiness scores in 2020 (mean={mean_2020:.3f}) were "
            f"{direction} than in 2019 (mean={mean_2019:.3f}). So it looks like the pandemic's start did shift global happiness levels."
        )
    else:
        interpretation_year = (
            f"p-value = {result_year.pvalue:.4f}, which is above {alpha}, (we fail to reject the null hypothesis) so this result is not "
            f"statistically significant. 2019 (mean={mean_2019:.3f}) and 2020 (mean={mean_2020:.3f}) "
            f"are close enough that the difference could just be normal year-to-year noise. "
            f"So based on this data, we can't say the start of the pandemic caused a noticeable change in how happy people said they felt."
        )
    logger.info(interpretation_year)

    # Compare 2 regions ---
    region_a = df[df["regional_indicator"] == "Western Europe"]["happiness_score"]
    region_b = df[df["regional_indicator"] == "Sub-Saharan Africa"]["happiness_score"]

    result_region = stats.ttest_ind(region_a, region_b, nan_policy="omit")

    mean_a = region_a.mean()
    mean_b = region_b.mean()

    logger.info(f"Western Europe mean happiness: {mean_a:.3f}")
    logger.info(f"Sub-Saharan Africa mean happiness: {mean_b:.3f}")
    logger.info(f"t-statistic: {result_region.statistic:.3f}")
    logger.info(f"p-value: {result_region.pvalue:.6f}")

    if result_region.pvalue < alpha:
        interpretation_region = (
            f"p-value = {result_region.pvalue:.6f}, way below {alpha}, (we reject the null hypothesis), so this difference is "
            f"definitely statistically significant. Western Europe (mean={mean_a:.3f}) and "
            f"Sub-Saharan Africa (mean={mean_b:.3f}) are clearly different in happiness scores. "
            f"This lines up with what I already noticed in the regional stats earlier makes sense given the huge gap in wealth, "
            f"health, and social support between these regions"
        )
    else:
            interpretation_region = (
            f"p-value = {result_region.pvalue:.6f}, above {alpha}, (we fail to reject the null hypothesis),  so I don't have enough evidence "
            f"that Western Europe (mean={mean_a:.3f}) and Sub-Saharan Africa (mean={mean_b:.3f}) "
            f"actually differ -- the gap I'm seeing could just be random."
        )
    logger.info(interpretation_region)

    return {
        "year_test": {
            "t_statistic": result_year.statistic,
            "p_value": result_year.pvalue,
            "mean_2019": mean_2019,
            "mean_2020": mean_2020,
            "interpretation": interpretation_year,
        },
        "region_test": {
            "t_statistic": result_region.statistic,
            "p_value": result_region.pvalue,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "interpretation": interpretation_region,
        },
    }
    
# Task 5: Correlation and Multiple Comparisons

@task(task_run_name="Correlation and Multiple Comparisons")
def correlation_multiple_comparison(df):
    logger = get_run_logger()
    numeric_df = df.select_dtypes(include="number")

    excluded_cols = ['happiness_score', 'year', 'ranking']
    explanatory_vars = [col for col in numeric_df.columns if col not in excluded_cols]
    results = {}
    for col in explanatory_vars:
        valid = numeric_df[[col, 'happiness_score']].dropna()
        logger.info(f"{col}: valid rows = {len(valid)}")
        coef, p_value = pearsonr(valid[col], valid['happiness_score'])
        results[col] = {
            "coefficient": coef, # r-coefficient
            "p_value": p_value #how can we trust this results 
        }
        logger.info(f"{col} r={coef:.3f}, p={p_value:.6f}")
    number_of_tests = len(explanatory_vars)
    alpha = 0.05 
    adjusted_alpha = alpha / number_of_tests
    
    logger.info(f"number of tests {number_of_tests}")
    logger.info(f"Adjusted alpha (Bonferroni correction): {adjusted_alpha}")
    
    for col, res in results.items():
        was_significant = res['p_value'] < alpha
        still_significant = res['p_value'] < adjusted_alpha
        logger.info(f"{col} significant at the original alpha = 0.05 {was_significant} | significant with adjusted_alpha: {still_significant}")
        
    return {
        "results": results,
        "number_of_tests": number_of_tests,
        "adjusted_alpha": adjusted_alpha
    }
    
# Task 6: Summary Report

@task(task_run_name="Summary Report")
def summary_report(df, stats_summary, hypothesis_result, correllation_result):
    logger =  get_run_logger()
    # Total number of countries and years in the merged dataset.
    total_countries = df['country'].nunique()
    total_years = df['year'].nunique()
    logger.info(f"Dataset covers {total_countries} unique countries across {total_years} years ")
    
    # The top 3 and bottom 3 regions by mean happiness score.
    by_region = stats_summary['by_region']
    top_3 = by_region.head(3)
    bottom_3 = by_region.tail(3)
    logger.info(f"Top 3 happinest regions:\n {top_3}")
    logger.info(f"Least 3 happy regions:\n {bottom_3}");
    # The result of the pre/post-2020 t-test in plain language.
    year_test = hypothesis_result['year_test']
    logger.info(f"2019 vs 2020 happiness comparison: {year_test['interpretation']}")

    # The variable most strongly correlated with happiness score (after Bonferroni correction).
    
    results = correllation_result['results']
    adjusted_alpha = correllation_result['adjusted_alpha']
    
    # should keep only variables that survived the Bonferroni correction
    significant_vars = {}
    for col, res in results.items():
        if res['p_value'] < adjusted_alpha:
            significant_vars[col] = res
    
    if significant_vars:
        strongest_var = max(significant_vars, key=lambda col: abs(significant_vars[col]['coefficient']))
        strongest_coef = significant_vars[strongest_var]['coefficient']
        strongest_p_value = significant_vars[strongest_var]['p_value']
        logger.info(
            f"Strongest variable correlated with happiness_score (after Bonferroni correction): "
            f"{strongest_var} (r={strongest_coef:.3f}, p={strongest_p_value:.6f})"
        )
    else:         
        logger.info("No variables remained significantly correlated with happiness_score after Bonferroni correction.")

    return {
        "total_countries": total_countries,
        "total_years": total_years,
        "top_3_region": top_3,
        "bottom_3_region": bottom_3,        
        "year_test_interpretation": year_test["interpretation"],
        "strongest_correlation": strongest_var if significant_vars else None,
    }
    
@flow
def happiness_pipeline(arr):
    raw_data = load_data(arr)
    combined = combine_data(raw_data)
    clean_df = clean_columns(combined)
    save_data(clean_df, output_path)
    stats_summary = compute_stats(clean_df)
    create_visualization(clean_df)
    hypothesis_result = hypothesis_test(clean_df)
    correlation_result = correlation_multiple_comparison(clean_df)
    final_summary = summary_report(clean_df, stats_summary, hypothesis_result, correlation_result)
    return final_summary

if __name__ == "__main__":
    happiness_pipeline(years)
    
    
