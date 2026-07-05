import pandas as pd
import numpy as np
from prefect import task, flow

@task(task_run_name="Create Series")
def create_series(arr):
    series = pd.Series(arr, name="values")
    return series
@task(task_run_name="Clean Data")
def clean_data(series):
    cleaned_series = series.dropna()
    return cleaned_series
@task(task_run_name="Summarize Data")
def summarize_data(series):
    return {
    "mean": series.mean(),
    "median": series.median(),
    "std": series.std(),
    "mode": series.mode()[0]
    }


@flow
def data_pipeline(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    result = summarize_data(cleaned_series)
    return result


arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

if __name__ == "__main__":
    result = data_pipeline(arr)
    for key, value in result.items():
        print(f"{key}: {value}")
    
        
    
# 1. Why might Prefect be more overhead than it is worth here?
# Since this pipeline is simple and only involves a few small functions, using Prefect may introduce unnecessary 
# complexity and overhead. Prefect is designed for orchestrating complex workflows, handling retries, scheduling,
# and monitoring tasks. For a straightforward data processing task like this, the added features of Prefect 
# may not provide significant benefits and could complicate 
# the codebase without improving performance or maintainability.

# 2. Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.
# In case of a simple pipeline, Prefect could still be useful in following scenarios:
# 1. Scheduling: If the data processing needs to be run at regular intervals (e.g., daily, weekly), Prefect can handle scheduling and ensure that the 
# pipeline runs automatically without manual intervention.
# 2. Monitoring and Logging: Prefect provides built-in monitoring and logging capabilities, which can be valuable for tracking the execution of the pipeline, 
# identifying issues, and maintaining a record of runs, even if the logic is simple.
# 3. Error Handling and Retries: If the pipeline is part of a larger system where failures can occur (e.g., network issues, temporary unavailability of data sources), 
# Prefect can manage retries and error handling, ensuring that the pipeline can recover from transient failures without manual intervention.
# 4. Integration with other services: If the pipeline needs to interact with other services or APIs 
# (e.g., fetching data from a database, sending notifications), Prefect can facilitate these integrations and manage dependencies between tasks, even if the core logic remains simple.