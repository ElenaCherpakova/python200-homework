from dotenv import load_dotenv
from openai import OpenAI
from pprint import pprint
import json



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

def get_completion(prompt: str, model="gpt-4o-mini", temperature=0):
    """
    Send a prompt to the model and return the assistant's text reply.
    This helper keeps our examples clean and focused on the prompt itself.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    return response.choices[0].message.content
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = """
Classify the sentiment of each review below as positive, negative, or mixed. Label each result with its review number.
"""
prompt += "\n".join([f"Review: {review}" for review in reviews])

response = get_completion(prompt)
print(f"Result: {response}")

# Q2
prompt = """
Classify the sentiment of each review below as positive, negative, or mixed. 
Label each result with its review number. Display in the following format: Example: 
Review: Fast shipping but the item arrived damaged." Sentiment: mixed
"""

prompt += "\n".join([f"Review: {review}" for review in reviews])
response = get_completion(prompt)
print(f"Result_2: {response}")

# Comments:
# The second prompt is more specific about the output format, which can help ensure that the model 
# provides the results in a consistent and expected manner. This is an example of prompt engineering, 
# where we refine the prompt to guide the model's output more effectively.

# Q3
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = """
Classify the sentiment of each review below as positive, negative, or mixed.
Label each result with its review number.
Examples:
Review: Super helpful, worth every penny. Sentiment: positive
Review: The product is terrible and broke on day one. Sentiment: negative
Review: Fast shipping but the item arrived damaged. Sentiment: mixed
Now classify these:
"""

prompt += "\n".join([f"Review: {review}" for review in reviews])
response = get_completion(prompt)
print(f"Result_3: {response}")

# Comments:
# The third prompt includes examples of how to classify sentiments, which can help the model understand the
# task better and produce more accurate results but in our case, I would go with the second prompt over third one because 
# it's not that necessary to provide examples for such a simple task. 
# The second prompt is more concise and still provides clear instructions for the model to follow.



# Q4
prompt = """
Show your step-by-step reasoning using plain text only, then give the final answer on its own line labelled: Final Answer: <value>
Problem: A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""
response = get_completion(prompt)
print(response)

# Comments:
# The prompt instructs the model to show its reasoning step-by-step, which can help in
# understanding how the model arrived at its final answer. This is particularly useful for complex problems
# where the reasoning process is important to verify the correctness of the answer. The final answer is
# clearly labeled, making it easy to identify the result of the calculations.

# Q5
review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

prompt = f"""
Classify the sentiment of the review and respond ONLY with valid JSON.
Return keys: sentiment (positive/negative/mixed), confidence (a float number from 0 to 1), reason (one short sentence).
Review: {review}.
"""
response = get_completion(prompt, temperature=0)
print("Raw response", response)

try:
    result = json.loads(response)
    print("Parsed sentiment:", result["sentiment"])
    print("Parsed confidence:", result["confidence"])
    print("Parsed reason:", result["reason"])
except json.JSONDecodeError:
    print("Error: Invalid JSON response")
    
# Q6

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

steps_prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

steps_response = get_completion(steps_prompt, temperature=0)
print("Steps response:", steps_response)

user_text_2 = "The weather today is sunny with a high of 75 degrees. Perfect day for a picnic!"
no_steps_prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."
```{user_text_2}```
"""
no_steps_response = get_completion(no_steps_prompt, temperature=0)
print(no_steps_response)

#Comments:
# Delimiters (triple backticks) help to clearly separate user instructions from data, reducing the risk of promp injection and misinterpretation.


# --- Local Models with Ollama ---

# Q1

# Ollama's comment:
# Thinking...
# Okay, the user wants me to explain a large language model in two 
# sentences. Let me start by breaking down the key elements. First, a large 
# language model is a type of artificial intelligence that can understand 
# and generate text. But I need to make sure I cover the main points without 
# getting too technical.

# I should mention that they have a vast amount of training data and use 
# complex algorithms to process information. Then, I can talk about how they 
# can create and understand text, like writing stories or answering 
# questions. Wait, but the user asked for two sentences. Let me check that 
# again. Yes, two sentences. Make sure each sentence is concise and covers 
# the essential aspects.
# ...done thinking.

# A large language model is a type of artificial intelligence designed to 
# understand and generate text, such as language, music, or any form of 
# human communication. It leverages massive datasets and advanced algorithms 
# to process and comprehend complex information, enabling it to create 
# meaningful content and perform tasks like answering questions or writing 
# stories.

prompt = """Explain what a large language model is in two sentences."""

response = get_completion(prompt)
print(f"OpenAI (gpt-4o-mini) response: {response}")

# OpenAi's comment:
#  A large language model is an artificial intelligence system designed to understand and 
#  generate human-like text by analyzing vast amounts of written data. It uses deep learning techniques, particularly neural networks, 
#  to predict and produce coherent and contextually relevant language based on the input it receives.


# Comments:
# Ollama responds with a more conversational and detailed explanation, while OpenAI's response is more concise and technical.
# Both provide accurate information, but the style and depth of explanation differ.

# Advantage and Disadvantage of running a model locally:
# Running a model locally can provide more control over the data and privacy, as well as potentially lower latency for certain applications. 
# However, it may require significant computational resources and technical expertise to set up and maintain,
# which can be a barrier for some users. Additionally, local models may not always have access to the 
# latest updates or improvements that cloud-based models receive, potentially leading to outdated performance or capabilities.
