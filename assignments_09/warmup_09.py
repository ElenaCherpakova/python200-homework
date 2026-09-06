# --- Supabase Connection ---
# Q1
# We need two pieces of information to connect to Supabase:
# - Project URL: identifies our Supabase project.
# - API Key: authenticates our requests.
#
# Supabase provides an anon (public) key and a service_role (secret) key.
# We will use the anon key for this exercise.
#
# API keys should never be hardcoded in source code. If the code is exposed,
# automated bots can quickly find and misuse the key. Instead, store API keys
# in environment variables.

# Q2
import os 
from dotenv import load_dotenv
from supabase import create_client


def get_client():
    load_dotenv()
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url:
        raise ValueError('SUPABASE_URL env is missing')
    if not key: 
        raise ValueError('SUPABASE_KEY env is missing')
    
    return create_client(url, key)

supabase = get_client()


# Q3
# Row Level Security (RLS) is a security feature that lets us define fine-grained access policies. 
# For example, we can allow users to read only their own rows. In production, RLS should be enabled to 
# prevent users from accessing or overwriting other users' data. During development, 
# it can add complexity, so we may temporarily disable it while testing data insertion and updates. 

# supabase-py CRUD
#Q1
record = {
    "date": "2026-08-24",
    "temperature_2m_max": 22.0,
    "temperature_2m_min": 16.0,
    "precipitation_sum": 0.0,
    "wind_speed_10m_max": 44.0,
}
def insert_test_record(supabase):
    response = supabase.table("weather_raw").insert(record).execute()
    return response.data
    
insert_test_record(supabase)

# Comment:
# If we run the function twice, the second insert will fail because "date"
# is the primary key and the record for 2026-08-24 already exists.
# The solution is upsert, which insert a new row if the key is new and update the existing row if it is not.
    
#Q2
def get_records_by_date_range(supabase, start, end):
    response = supabase.table("weather_raw").select('*').gte('date', start).lte('date', end).execute()
    return response.data

print(get_records_by_date_range(supabase, "2026-08-23", "2026-08-25"))

#Q3
# insert adds new rows and will fail if a row with the same primary/unique key already exists.
# Upsert inserts a new row or updates the existing row if there is a conflict.
# Example: we would use insert when we know the records are new. Use upsert when
# loading weather data that may already exist for the same date.
def safe_upsert(supabase, records):
    response = supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
    print(f"Rows affected: {len(response.data)}")
    return response.data

safe_upsert(supabase, [record])

# Idempotency
#Q1
# Idempotency is a key quality of robust data pipelines because it allows
# the pipeline to be safely restarted without creating duplicate data.
# For example, if a pipeline inserts 500 weather records and crashes after
# inserting 300, restarting the pipeline may insert those 300 records again.
# This can create duplicate rows or cause errors if those rows have unique keys.
