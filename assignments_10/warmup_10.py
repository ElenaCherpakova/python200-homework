# Part 1: Warmup

# Q1
# The ML model makes the `good` or `skip` prediction using the numbers and features in the data. 
# It is good for this because it is trained to find patterns in structured data. 
# The LLM takes the results and writes a recommendation in normal language. 
# It is good for this because it can understand and generate text.
# If we switched them, the results would not be as good. 
# The LLM could make the `good` or `skip` prediction, but it would be slower and less consistent for this simple task. 
# The ML model could make a prediction, but it cannot easily write a natural-language recommendation. 
# So, each model is used for the task it is best at.

# Q2

# Converting a date string like "2023-07-04" to day-of-week - deterministic code because converting a date is simple, fixed rule that code can do quickly and accurately.
# Classifying a job posting as "entry-level", "mid-level", or "senior" based on freeform text - LLM, because it can understand freeform text and make nuanced classifications based on context and language.
# Predicting customer churn given 15 numeric features and a labeled training dataset - ML model, because it can learn patterns from structured numeric data and make predictions based on those patterns.
# Normalizing inconsistent city names ("NYC", "New York City", "New York, NY") to a canonical form - deterministic code, because it can use a fixed mapping or rules to standardize the names quickly and accurately.
# Summing a column of revenue figures - deterministic code, because it is a straightforward arithmetic operation that can be performed quickly and accurately by code.

# Q3
# Incremental processing means processing only the new data since the last run instead of processing all the data again. 
# It is important for this pipeline because it saves time and reduces the cost of using the LLM. 
# If the transform script processed all 365 records every time, it would repeatedly process the same records, 
# which would increase the cost and could also create duplicate or inconsistent results.

# Prompt Design
# Q1
    # "You are writing a two-sentence running recommendation for a daily weather summary app. "
    # "The first sentence should state the prediction, and the second sentence should explain the reasoning."
    # "You will receive weather conditions for a single day and a machine learning prediction "
    # "about whether the day is good for running. "
    # "Write exactly two sentences - direct, practical, and specific to the conditions. "
    # "Do not use bullet points, headers, or phrases like 'Based on the data'."

# For validation, I would change the logic from checking for exactly one sentence to checking for exactly two sentences.
# I would also make sure the first sentence contains the prediction and the second sentence gives the reasoning.


# Q2
import time
def call_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries + 1):
        try: 
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            raw_summary = response.choices[0].message.content
            return raw_summary
        except Exception as e:
            print(f"Error occurred: {e}")
            if attempt < max_retries:
                print(f"Retrying... ({max_retries - attempt} retries left)")
                time.sleep(2)
            else:
                print("Max retries reached. Returning None.")
                return None
            
# I would use this in a production pipeline when calling an LLM API because
# temporary errors or network problems can happen. Retrying can help the
# pipeline recover from temporary failures instead of stopping completely.