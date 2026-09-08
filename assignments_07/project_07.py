from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from scipy.stats import pearsonr
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
api_key = os.getenv('OPEN_API_KEY')
print('OpenAI client created.')

# Pre-task: Load the Data
# DATA_PATH = "../assignments_01/outputs/merged_happiness.csv"
DATA_PATH = "assignments_01/outputs/merged_happiness.csv"
# Task 1: Define Your Tools
# data_dir = "../assignments_01/happiness_project"
data_dir = "assignments/resources/happiness_project"
file_name = "world_happiness"
# print(os.getcwd())  # confirm what directory you're actually running from
# print(os.path.exists(DATA_PATH))  # True/False for the merged file
years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

df = None

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into the shared global DataFrame.

    Loads the merged CSV from DATA_PATH. If it does not exist, falls back
    to loading and merging the yearly CSV files.

    Returns:
        A dict containing "shape" and "columns", or an "error" key.
    """
    global df

    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
        else:
            res = []

            for year in years:
                file_path = os.path.join(
                    data_dir,
                    f"{file_name}_{year}.csv"
                )

                try:
                    year_df = pd.read_csv(
                        file_path,
                        sep=';',
                        decimal=',',
                        encoding='utf-8'
                    )
                    year_df['year'] = year
                    res.append(year_df)

                except FileNotFoundError:
                    return {"error": f"File {file_path} not found."}

            if not res:
                return {"error": "No yearly data files were loaded."}

            df = pd.concat(res, ignore_index=True)

    except Exception as e:
        return {"error": f"Failed to load data: {str(e)}"}

    return {
        "shape": df.shape,
        "columns": list(df.columns)
    }
    
@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.
    
    Args:
        column: the name of the column to summarize
    
    Returns:
        a dict of descriptive statistics (from pandas .describe()), or 
        a dict with an "error" key if no data is loaded or the column doesn't exist. 
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    else:
        return df[column].describe().to_dict()
    
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.
    Args:
        col1: Name of the first column
        col2: Name of the second column
        
    Returns:
        A dict with "col1", "col2", "pearson_r", and "p_value", or a dict
        with an "error" key on bad input.
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    
    missing_cols = []
    if col1 not in df.columns:
        missing_cols.append(col1)
    if col2 not in df.columns:
        missing_cols.append(col2)
            
    if missing_cols: 
        return {"error": f"These columns are not in the data: {', '.join(missing_cols)}"}

        # clean missing (NaN) values
    x = df[col1]
    y = df[col2]
    valid_mask = x.notna() & y.notna()
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
        
        # check if we have enough data point (Pearson requies at least 2 points)
    if(len(x_clean) < 2):
        return {
            "error": "Not enough valid paired data points to compute correlation"
        }
        # compute Pearson correlation safely
    try: 
        res = pearsonr(x_clean, y_clean)
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(res.statistic), 4),
            "p_value": round(float(res.pvalue), 4)
        }
    except Exception as e:
        return {"error": f"Failed to compute correlation: {(str(e))}"}

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.
    
    Args:
        column: Name of the column to rank countries by (e.g. 'happiness_score').
        year: The year to filter the data on.
        n: Number of top countries to return (default 5).
    
    Returns:
        A list of dicts, each with "country" and the requested column's
        value, sorted descending and limited to the top n rows.
        Returns a dict with an 'error' key on bad input.
    """
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    if 'country' not in df.columns:
        return {"error": "Column 'country' not found in the data."}
    if year not in df['year'].unique():
        return {"error": f"Year {year} not found in the data."}
    if n <= 0: 
        return {"error": "n must be a positive integer."}
    filtered_df = df[df['year'] == year].sort_values(by=column, ascending=False).head(n)
    return filtered_df[['country', column]].to_dict(orient='records')

# Task 2: Build the Agent
model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.

Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries.

For custom analysis or plots the tools don't cover (e.g. grouping by year and region),
a variable named `df` containing the full dataset will be available directly in your
code execution environment — use it as a regular pandas DataFrame.

Use pd.concat() to combine DataFrames, not .append() (removed in modern pandas).

Write code directly only when the tools aren't sufficient.
Be concise and student-friendly in your responses.
"""


agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

# Running the Project

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    
    
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False, additional_args={"df": df})
        print(response)

    # My query 1
    my_query_1 = "What's the correlation between social_support and happiness_score?"  
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: Did this trigger tool use, code generation, or both?
    # Triggered tool use only. The agent called compute_correlation(col1='social_support',
    # col2='happiness_score') directly in Step 1, got the result (pearson_r=0.7439, p_value=0.0),
    # then called final_answer() in Step 2 with an interpretation. No raw code or custom logic
    # was needed since this mapped exactly onto an existing tool.
    # My query 2
    my_query_2 = "Create a bar chart comparing the top 5 and bottom 5 countries by happiness_score in 2022."   
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: Did this trigger tool use, code generation, or both?
    # Triggered both tool use and code generation. The agent called get_top_n_countries()
    # to retrieve country/score data, then wrote its own matplotlib code (plt.bar,
    # color-coding top vs. bottom, saving the figure) since no tool covers bar chart
    # generation. This query required several iterations to get right — the agent initially tried
    # calling get_top_n_countries() twice with identical arguments to get both "top" and "bottom"
    # results, then self-corrected by fetching all countries and sorting locally. It also hit and
    # recovered from an internal dict-unpacking syntax restriction in the sandboxed interpreter.

# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
# significant? Did it use the p-value correctly? What threshold did it apply?
# The agent reported a Pearson correlation of 0.6313 and a p-value of 0.0, and concluded that
# the correlation was statistically significant.
# It used the p-value correctly because it was below the usual 0.05 significance threshold.
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
# you expected, or less? Describe one specific example.
# Less capable: when asked to plot happiness_score by region, the agent kept trying to
# access a variable called df directly, even though that variable doesn't exist in its
# code execution environment by default. It took several failed attempts (and even a
# hallucinated import) before I fixed this by passing df into the agent's execution
# context directly via additional_args={"df": df} on each agent.run() call.
# More capable: when a plot failed due to a macOS-specific threading error, the agent
# diagnosed the issue on its own and added matplotlib.use('Agg') without being told.

# 3. What one additional tool would make this agent meaningfully more useful?
# Describe what it would do and what kind of question it would help the agent answer.
# (You do not need to implement it.)
# A get_yearly_trend(country, column) tool that returns a column's value for one country
# across all years. This would let the agent answer trend questions directly, like
# "how has Finland's happiness_score changed since 2015?", without needing to fetch the
# whole dataset and filter it manually in code.