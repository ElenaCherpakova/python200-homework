# link: https://drive.google.com/file/d/1aq-Z7sYdVoTja12C6mJYZfLrftHcpln3/view?usp=sharing

import os
from dotenv import load_dotenv
from prefect import flow, task
import requests
import joblib
import json
import pandas as pd
from supabase import create_client
from openai import OpenAI



load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
print("Key loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("Key prefix:", os.getenv("OPENAI_API_KEY", "")[:7])

# Extract task

@task(retries=2, retry_delay_seconds=10)
def extract() -> list[dict]:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
    "latitude": 55.7558,
    "longitude": 37.6173,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "Europe/Moscow",
}
    response = requests.get(url, params=params)
    response.raise_for_status() 
    
    df = pd.DataFrame(response.json()["daily"])
    # Changes time into our db date field:
    df["date"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")  
    df = df.drop("time", axis=1)  
    # Converts the DataFrame into a list of row dictionaries:
    records = df.to_dict(orient="records")
    print(f"Extracted {len(records)} weather records")
    return records
    

# load_raw task
@task(retries=2, retry_delay_seconds=5)     
def load_raw(table_name, data: list[dict]) -> list:
    
    response = supabase.table(table_name).upsert(data, on_conflict="date").execute()
    print(f"Upserted {len(response.data)} raw records into weather_raw")
    return response.data

# transform task

SYSTEM_PROMPT = (
    "You are writing a one-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly one sentence — direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)
def make_user_message(row, good_for_running, confidence):
    prediction_text = "good for running" if good_for_running else "not ideal for running"
    return (
        f"Date: {row['date']}\n"
        f"High: {row['temperature_2m_max']}°C, Low: {row['temperature_2m_min']}°C\n"
        f"Precipitation: {row['precipitation_sum']} mm\n"
        f"Max wind speed: {row['wind_speed_10m_max']} km/h\n"
        f"Model prediction: {prediction_text} (confidence: {confidence:.0%})"
    )
def validate_summary(text):
    text = text.strip()
    if not text:
        return None
    sentences = [s for s in text.split(".") if s.strip()]
    if len(sentences) > 1:
        return None
    return text
@task(retries=2, retry_delay_seconds=5)
def transform(table_name, raw_records):
    enrichment_response = supabase.table(table_name).select("date").execute()
    already_done = {row["date"] for row in enrichment_response.data}
    to_classify = [row for row in raw_records if row["date"] not in already_done]
    print(f"Records to process: {len(to_classify)} (skipping {len(already_done)} already enriched)")
    
    if not to_classify:
        print("Nothing to do — all records already enriched.")
        return []
    
    with open("models/weather_classifier_metadata.json") as f:
        metadata = json.load(f)
    FEATURES = metadata["features"]
    df = pd.DataFrame(to_classify)
    X = df[FEATURES]
    
    clf = joblib.load("models/weather_classifier.pkl")
    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)[:, 1]
    print(f"Good days predicted: {predictions.sum()} / {len(predictions)}")
    print(f"Confidence range: {probabilities.min():.2f} – {probabilities.max():.2f}")
    
    enrichment_records = [
    {
        "date":             to_classify[i]["date"],
        "good_for_running": bool(predictions[i]),
        "confidence":       round(float(probabilities[i]), 4),
        "llm_summary":      None,
    }
    for i in range(len(to_classify))
]
    for i, record in enumerate(enrichment_records):
        raw_row = next(r for r in to_classify if r["date"] == record["date"])
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": make_user_message(
                            raw_row,
                            record["good_for_running"],
                            record["confidence"],
                        ),
                    },
                ],
                max_tokens=100,
            )
            raw_summary = response.choices[0].message.content
            summary = validate_summary(raw_summary) or "Recommendation unavailable"
        except Exception as e:
            print(f"  API error on {record['date']}: {e}")
            summary = "Recommendation unavailable."

        record["llm_summary"] = summary
        if(i + 1) % 50 == 0:
            print(f"  Enriched {i + 1} / {len(enrichment_records)} records...")
        
    return enrichment_records

# load_enriched task
@task(retries=2, retry_delay_seconds=5)
def load_enriched(table_name, enrichment_records) -> list:
    if not enrichment_records:
        print("Nothing to load")
        return []
    
    df_response = supabase.table(table_name).upsert(enrichment_records, on_conflict="date").execute()
    print(f"Upserted {len(df_response.data)} rows into {table_name}")

    return df_response.data
    


# Flow
@flow(log_prints=True)
def etl_pipeline():     
    raw_data = extract()
    load_raw("weather_raw", raw_data)
    transformed_data = transform("weather_enriched", raw_data)
    enriched_data = load_enriched("weather_enriched", transformed_data)
    print("ETL pipeline completed successfully.")

    
if __name__ == "__main__":
    etl_pipeline()