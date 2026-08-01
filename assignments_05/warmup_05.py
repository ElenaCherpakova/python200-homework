from dotenv import load_dotenv
from openai import OpenAI
from pprint import pprint


# --- Completions API ---

# API Q1

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print(f"Text response: {response.choices[0].message.content}")
print(f"Total tokens used: {response.usage.total_tokens}")
print(f"Model used: {response.model}")

# API Q2

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response =  client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=temp
)
    print(f"Temperature: {temp}")
    print(f"Text response for {temp}: {response.choices[0].message.content}")
    
#Comments:
# For temperature 0 and 0.7, the responses are more focused and coherent, while at temperature 1.5, 
# the response is more creative and diverse, but may also be less relevant or coherent.

# API Q3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

print("Responses for API Q3:")
for i, choice in enumerate(response.choices, start=1):
    print(f"Response {i}: {choice.message.content}")

# API Q4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)
print(f"Text response for API Q4: {response.choices[0].message.content}")

#Comments:
# We might want to use max_tokens parameter to limit the length of the response.
# This can be particularly useful in applications where response length is critical, 
# such as in chatbots or when displaying information on limited screen space.
# In our case, we set max_tokens to 15, which means the model will generate a response that is at most 15 tokens long and 
# may be cut off if the explanation requires more tokens to be complete.

# --- System Messages and Personas ---
# Q1
print("\n--- System Messages and Personas Q1 ---")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}]
)

print(f"Text response for System Messages and Personas Q1: {response.choices[0].message.content}")

response_2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
    {"role": "system", "content": "You are a professor of Python course at university. You explain concepts clearly and provide academic insights."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}]
)
print(f"Text response for System Messages and Different Personas: {response_2.choices[0].message.content}")

# Comments:
# The first response is more encouraging and simplified, while the second response is more academic and detailed.

# Q2
print("\n--- System Messages and Personas Q2 ---")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages =messages
)

print(f"Text response for System Messages and Personas Q2: {response.choices[0].message.content}")
# Comments:
# The model is able to remember the user's name from the previous messages and respond accordingly.


# --- Prompt Engineering ---
# Q1

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = "Classify the sentiment of each review below as positive, negative, or mixed. Label each result with its review number.\n\n"
prompt += "\n".join([f"Review: {review}" for review in reviews])

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [{"role": "user", "content": prompt}]
)
print(f"Result: {response.choices[0].message.content}")

# Q2
prompt = 'Classify the sentiment of each review below as positive, negative, or mixed. Label each result with its review number. Display in the following format: Example: Review: "Fast shipping but the item arrived damaged."Sentiment: mixed\n\n'
prompt += "\n".join([f"Review: {review}" for review in reviews])
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [{"role": "user", "content": prompt}]
)
print(f"Result_2: {response.choices[0].message.content}")
# Comments:
# The second prompt is more specific about the output format, which can help ensure that the model 
# provides the results in a consistent and expected manner. This is an example of prompt engineering, 
# where we refine the prompt to guide the model's output more effectively.