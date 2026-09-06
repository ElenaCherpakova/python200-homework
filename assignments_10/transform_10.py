# Step 1: Incremental Read

import json
import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
import joblib
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)


supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
response = supabase.table("weather_raw").select("*").execute();
raw_rows = response.data

enriched_response = supabase.table("weather_enriched").select("date").execute();
already_done = {row["date"] for row in enriched_response.data}

to_classify = [row for row in raw_rows if row["date"] not in already_done]

print(f"Raw records: {len(raw_rows)}") # 366
print(f"Already enriched: {len(already_done)}") # 0
print(f"Will be processed: {len(to_classify)}") # 366

# Step 2: ML Transform

FEATURES = metadata["features"]
df = pd.DataFrame(to_classify)

X = df[FEATURES]


clf = joblib.load("models/weather_classifier.pkl")

predictions = clf.predict(X)
probabilities = clf.predict_proba(X)[:, 1]  
print(f"Good days predicted: {predictions.sum()} / {len(predictions)}")
print(f"Confidence range: {probabilities.min():.2f} – {probabilities.max():.2f}")

enrichment_records = []
for i, row in enumerate(to_classify):
    enrichment_records.append({
        "date":  row["date"],
        "good_for_running": bool(predictions[i]),
        "confidence": round(float(probabilities[i]), 4)
    })

print("Sample enrichment records:")
for r in enrichment_records[:3]:
    print(r)

# Sample enrichment records:
# {'date': '2026-08-24', 'good_for_running': True, 'confidence': 0.7686}
# {'date': '2023-01-01', 'good_for_running': False, 'confidence': 0.0809}
# {'date': '2023-01-09', 'good_for_running': False, 'confidence': 0.1918}

# Sanity Check:
good_days = [r for r in enrichment_records if r["good_for_running"]]
skip_days = [r for r in enrichment_records if not r["good_for_running"]]

print(f"Good days: {len(good_days)} ({len(good_days)/len(enrichment_records):.0%})")
print(f"Skip days: {len(skip_days)}")

# Show a few high-confidence and borderline predictions
enrichment_records.sort(key=lambda r: r["confidence"], reverse=True)
print("\nHigh-confidence good days for running:")
for r in enrichment_records[:3]:
    print(f" {r['date']}: {r['confidence']:.3f}")

enrichment_records.sort(key=lambda r: abs(r["confidence"] - 0.5))
print("\nMost borderline (closest to 0.5 confidence):")
for r in enrichment_records[:3]:
    print(f" {r['date']}: {r['confidence']:.3f}")

# Good days: 128 (35%)
# Skip days: 238

# High-confidence good days for running:
#  2023-09-04: 0.947
#  2023-09-06: 0.941
#  2023-07-28: 0.925

# Most borderline (closest to 0.5 confidence):
#  2023-05-18: 0.505
#  2023-11-03: 0.516
#  2023-05-25: 0.521

# Step 3: LLM Transform

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
    # Reject if more than two sentences 
    sentences = [s for s in text.split('.') if s.strip()]
    if len(sentences) != 1:
        return None
    return text

for i, record in enumerate(enrichment_records):
    try:
        raw_row = next(r for r in to_classify if r["date"] == record["date"])
    
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                "content": make_user_message(
                    raw_row,
                    record["good_for_running"],
                    record["confidence"],
                    )
                }
            ],
            max_tokens=100,
        )
        raw_summary = response.choices[0].message.content
        summary = validate_summary(raw_summary) or "Recommendation unavailable."

    except Exception as e:
        print(f"Error processing record {record['date']}: {e}")
        summary = "Recommendation unavailable."
        
    record["llm_summary"] = summary
        
    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1} / {len(enrichment_records)} records")
        
    
# Step 4: Load

response = (
    supabase.table("weather_enriched")
    .upsert(enrichment_records, on_conflict="date")
    .execute()
)

print(f"Upserted {len(response.data)} rows into weather_enriched")

# Step 5: Verify

# Step 5: Verify

# Get total number of rows
response = supabase.table("weather_enriched").select("*").execute()
all_rows = response.data

print(f"Total rows: {len(all_rows)}")

# Count good days
good_days = sum(row["good_for_running"] for row in all_rows)
print(f"Good days: {good_days}")

# Print five sample rows
print("\nSample rows:")
for row in all_rows[:5]:
    print(
        f"\n{row["date"]} | "
        f"good={row["good_for_running"]} | "
        f"conf={row["confidence"]:.2f}"
    )
    print(f"  {row["llm_summary"]}")
    
    
# I looked at several of the LLM summaries. Most of the summaries that were generated accurately
# reflected the weather and the model's prediction. A good example was the May 25 summary 
# because it mentioned the mild temperature and wind and agreed with the model's `good=True` 
# prediction. A weaker example was May 18 because the recommendation was unavailable even 
# though the model predicted that it was a good day for running. 
# This could have been caused by an API error or the LLM response not passing the validation check.


# Step 6: Reflect
# I used weather data from Toronto, which is different from the data the ML classifier was trained on, 
# so its predictions may not be as accurate.
# The model may have learned weather patterns that do not apply as well to Toronto.
# The LLM cannot override the classifier because it uses the classifier's prediction and weather features
# to create a recommendation. It is mainly adding a natural-language explanation to the ML prediction. 
# If I ran the pipeline on 50,000 records, my main concerns would be cost and processing time 
# because of the number of LLM calls. I would handle this by processing the data incrementally 
# and only sending new records to the LLM.
