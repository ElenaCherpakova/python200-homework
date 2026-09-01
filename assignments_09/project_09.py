# Link to the video: https://drive.google.com/file/d/1dgoWPioB9CtTViRhA6kYpse4PBudLxQU/view?usp=drive_link

import requests

# Step 1: Extract
print("\n---------Step 1: Extract---------\n")
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 43.65,
    "longitude": -79.38,
    "start_date": "2023-01-01",
    "end_date":   "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}

response = requests.get(url, params=params)
response.raise_for_status() # raise_for_status() - raises exception immediately if the server returned a 4xx or 5xx response.
data = response.json()
print("Summary of the response")
print(f"Location: lat: {data['latitude']}, long: {data['longitude']}")
print(f"Date range: from {data['daily']['time'][0]} to {data['daily']['time'][-1]}")
print(f"Number of daily records: {len(data['daily']['time'])}")

print("\n---------Step 2: Transform---------\n")
# Step 2: Transform
daily = data['daily']

records = [
    {
        "date":  daily["time"][i],
        "temperature_2m_max": daily["temperature_2m_max"][i],
        "temperature_2m_min": daily["temperature_2m_min"][i],
        "precipitation_sum": daily["precipitation_sum"][i],
        "wind_speed_10m_max": daily["wind_speed_10m_max"][i]
        
    }
    for i in range(len(daily["time"]))
]

print(f"Prepared {len(records)} records")
print("First records:", records[0])
print("Last records:", records[-1])
#Comments:
# A full year in 2023 should contain 365 records because 2023 was not a leap year. 
# If the numbers differ, the API may have missing dates, 
# or the requested date range/timezone may be incorrect.

print("\n---------Step 3: Load---------\n")
# Step 3: Load

import os 
from dotenv import load_dotenv
from supabase import create_client
from datetime import date
load_dotenv()

supabase = create_client(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_KEY")
)

response = supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
print(f"Upserted {len(response.data)} rows into weather_raw table")

#Comments:
# The row count should remain the same because records with the same
# date are updated rather than inserted as duplicates.
# This shows that the pipeline is idempotent: running it multiple times
# produces the same final state in the database.

print("\n---------Step 4: Verify---------\n")
# Step 4: Verify

#Row count
count_response = supabase.table("weather_raw").select("date", count="exact").execute()
print(f"Rows in weather_raw: {count_response.count}")
# Spot-check: first and last record
first = supabase.table("weather_raw").select("*").eq("date", "2023-01-01").execute()
last  = supabase.table("weather_raw").select("*").eq("date", "2023-12-31").execute()
# july = supabase.table("weather_raw").select("*").eq("date", "2023-07-04").execute()

# nearest 
target_date = date(2023, 7, 4)
response = supabase.table("weather_raw").select("*").execute()
rows = response.data
nearest = min(
    rows, 
    key=lambda row: (
        abs(date.fromisoformat(row['date']) - target_date), 
        date.fromisoformat(row['date'])
        )
    );

print("Earliest record:", first.data)
print("Latest record: ", last.data)
# print("July 4th record:", july.data)
print("Nearest record to July 4th:", nearest)

# Rows in weather_raw: 365
# Earliest record: [{'date': '2023-01-01', 'temperature_2m_max': 3.5, 'temperature_2m_min': 1.9, 'precipitation_sum': 1.8, 'wind_speed_10m_max': 18.1, 'loaded_at': '2026-09-01T01:21:28.765864+00:00'}]
# Latest record:  [{'date': '2023-12-31', 'temperature_2m_max': 1.7, 'temperature_2m_min': -0.6, 'precipitation_sum': 2.0, 'wind_speed_10m_max': 13.6, 'loaded_at': '2026-09-01T01:21:28.765864+00:00'}]
# Nearest record to July 4th: [{'date': '2023-07-04', 'temperature_2m_max': 28.7, 'temperature_2m_min': 17.9, 'precipitation_sum': 0.1, 'wind_speed_10m_max': 16.2, 'loaded_at': '2026-09-01T01:21:28.765864+00:00'}]
