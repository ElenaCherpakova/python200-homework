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
    """Load the World Happiness dataset.

    Loads the merged CSV if it exists. Otherwise, loads and merges
    the yearly CSV files from the happiness_project resources folder.

    Returns:
        A dict containing the dataset shape and column names,
        or an error message.
    """
    global df

    try:
        # First try the merged dataset
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)

        # Otherwise fall back to yearly files
        else:
            yearly_files = sorted(
                [
                    filename
                    for filename in os.listdir(data_dir)
                    if filename.startswith("world_happiness_")
                    and filename.endswith(".csv")
                ]
            )

            if not yearly_files:
                return {"error": f"No yearly CSV files found in {data_dir}."}

            frames = []

            for filename in yearly_files:
                file_path = os.path.join(data_dir, filename)

                year_df = pd.read_csv(
                    file_path,
                    sep=";",
                    decimal=",",
                    encoding="utf-8"
                )

                year = int(filename.replace("world_happiness_", "").replace(".csv", ""))
                year_df["year"] = year
                frames.append(year_df)

            df = pd.concat(frames, ignore_index=True)

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

# Running the Project

def main():
    os.makedirs("outputs", exist_ok=True)

    model = OpenAIServerModel(
        api_key=api_key,
        model_id="gpt-4o-mini"
    )

    agent = CodeAgent(
        tools=[
            load_happiness_data,
            summarize_column,
            compute_correlation,
            get_top_n_countries
        ],
        model=model,
        instructions=SYSTEM_PROMPT,
        additional_authorized_imports=[
            "pandas",
            "matplotlib.pyplot",
            "scipy.stats"
        ],
        max_steps=8,
    )

    # Task 3: Guided queries
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

    # Task 4: Custom Query 1
    my_query_1 = "What's the correlation between social_support and happiness_score?"
    response_1 = agent.run(my_query_1, reset=False)
    print(f"\n--- Custom Query 1 ---")
    print(response_1)

# Observation: The agent used the predefined compute_correlation tool
# because this calculation was directly supported by an available tool.
# No custom Python code was needed for this query.

    # Task 4: Custom Query 2
    my_query_2 = (
        "Create a bar chart comparing the top 5 and bottom 5 countries "
        "by happiness_score in 2022."
    )

    response_2 = agent.run(my_query_2, reset=False)
    print(f"\n--- Custom Query 2 ---")
    print(response_2)
    
# Observation: The agent used the get_top_n_countries tool to retrieve
# the top countries, then generated and executed Python/matplotlib code
# to create the bar chart because the available tool did not directly
# provide a top-and-bottom comparison chart.


if __name__ == "__main__":
    main()
    
# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
# significant? Did it use the p-value correctly? What threshold did it apply?
#
# The agent reported a Pearson correlation of 0.6313 and a p-value of 0.0.
# It correctly concluded that the correlation was statistically significant
# because the p-value was below the standard 0.05 threshold.
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
# expected, or less? Describe one specific example.
#
# I was surprised that the CodeAgent could generate custom matplotlib code when
# the available tools did not directly support the requested plot. This made it
# more flexible than a ToolCallingAgent for custom analysis and visualization tasks.
#
# 3. What one additional tool would make this agent meaningfully more useful?
# Describe what it would do and what kind of question it would help the agent answer.
#
# A get_yearly_trend(country, column) tool would return a selected column's
# values for one country across all years. It would help answer questions such
# as "How has Finland's happiness_score changed since 2015?" without requiring
# the agent to manually filter the full dataset.