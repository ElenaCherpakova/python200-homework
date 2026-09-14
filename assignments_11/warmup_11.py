# Prefect Orchestration

# Q1
# A @task is a individual unit of work, while @flow is the collection of tasks that make up workflow. 
# It defines the dependencies between tasks and the order in which they need to be executed.

# I would not decorate a simple Celsius-to-Fahrenheit helper with @task
# because it is a pure, in-memory calculation with no I/O or other
# orchestration needs. It can simply be a regular Python function.

# Q2
from prefect import flow, task
@task(retries=3, retry_delay_seconds=30)
def call_api():
    pass
    
# Q3

# I would look at the flow run details in the Prefect UI
# and open the failed `transform` task.
# I would expect to find the error message, traceback/logs,
# task state, and information about when and why the task failed.
# Since `transform` failed, `load_enriched` did not run because
# it depends on the successful completion of `transform`.

# Production Patterns

# Q1
# raise_for_status() checks the HTTP response and raises an exception
# when the API returns a 4xx or 5xx error.This is better than `if response.status_code != 200: print("error")`
# because printing an error does not stop the task. The pipeline may
# continue with bad or incomplete data. When the API returns a 500 error, raise_for_status() raises an exception,
# so Prefect marks the task as Failed and downstream tasks that depend on
# it will not run. With only `print("error")`, the task can still be marked Completed,
# so downstream tasks may run with bad data. This can lead to corrupted data or misleading results.

# Q2
# Upsert protect us from duplicate records when the pipeline is re-run.
# With on_conflict="date", existing rows with the same date are updated instead of causing a duplicate-key error.
# If we used plain insert, the existing dates would cause a duplicate-key error, and the pipeline could fail
# during the load_raw step.

# Q3
from prefect.logging import get_run_logger

@task
def load_enriched(enrichment_records: list) -> None:
    logger = get_run_logger()
    logger.info(f"Number of enrichment records upserted: {len(enrichment_records)}")

# Q4
# The incremental processing check contributes to idempotency by ensuring
# that records that have already been enriched are skipped. Running the
# pipeline again therefore does not repeatedly process the same records.
# Without this check, all 365 records would go through the ML and LLM
# steps every time. This would increase API/LLM costs, make the pipeline
# take longer, and repeatedly process data that has already been enriched.
# It could also cause unnecessary overwrites or inconsistent results if
# the ML/LLM output changes between runs.