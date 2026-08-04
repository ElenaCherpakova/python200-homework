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
    

# Comments:
# What I notice: at temperature=0, the output is deterministic and predictable -
# running it again gives the same or nearly the same name every time. At
# temperature=0.7, the output is somewhat more varied while still sounding like
# a plausible, coherent business name. At temperature=1.5, the output becomes
# noticeably more unusual and creative, but also risks sounding less like a
# real company name and more like a random or slightly incoherent phrase.
#
# Which temperature for consistent, reproducible output: temperature=0. It
# minimizes randomness in the model's token selection, so repeated calls with
# the same prompt return the same (or very nearly the same) result -- exactly
# what's needed when reproducibility matters more than creative variation.

# API Q3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)
for i, choice in enumerate(response.choices, start=1):
    print(f"Completion {i}: {choice.message.content}")
    

# API Q4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)
print(f"Text response for API Q4: {response.choices[0].message.content}")

# Comments:
# What happened: the response is cut off mid-sentence/mid-thought, because
# max_tokens=15 caps generation at 15 tokens - far too few to fully explain
# neural networks, so the output is TRUNCATED rather than a complete, shorter
# explanation the model chose on its own.
# Why use max_tokens in a real application: it controls API cost (fewer
# tokens billed per call) and enforces hard length limits for space-constrained
# UI, like a chat bubble, notification, or preview card - useful even though
# it risks cutting a response off before the full idea is expressed.

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
# The first response (encouraging tutor persona) is simpler, warmer, and ends
# with explicit encouragement, matching the system prompt's instruction.
# The second response (professor persona) is more formal and academic, using
# more precise terminology and giving broader context about why list
# comprehensions exist, rather than just how to use them. Same question,
# same underlying model, but the system message meaningfully shapes tone,
# vocabulary, and depth.

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
# The model correctly recalls "Jordan" from the earlier user turn in the
# messages list. This works because the full conversation history -- including
# both the user's and the assistant's prior turns -- is sent with every API
# call; the model itself has no memory between calls, so recall only happens
# because the calling code includes the full history each time.

# --- Prompt Engineering ---

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

# Q1 -- Zero-shot (no examples given)
prompt = """
Classify the sentiment of each review below as positive, negative, or mixed. Label each result with its review number.
"""
prompt += "\n".join([f"{i+1}. Review: {review}" for i, review in enumerate(reviews)])

response = get_completion(prompt)
print(f"Zero-shot result: {response}")

# Q2 -- One-shot (a single example showing the desired output format)

prompt = """
Classify the sentiment of each review below as positive, negative, or mixed. 
Label each result with its review number. Display in the following format: Example: 
Review: Fast shipping but the item arrived damaged." Sentiment: mixed
"""

prompt += "\n".join([f"{i+1}. Review: {review}" for i, review in enumerate(reviews)])
response = get_completion(prompt)
print(f"One-shot result: {response}")

# Comments:
# The One-shot result is more specific about the output format, which can help ensure that the model 
# provides the results in a consistent and expected manner. This is an example of prompt engineering, 
# where we refine the prompt to guide the model's output more effectively.

# Q3 -- Few-shot (multiple examples, one per class: positive, negative, mixed)

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

prompt += "\n".join([f"{i+1}. Review: {review}" for i, review in enumerate(reviews)])
response = get_completion(prompt)
print(f"Few-shot result: {response}")

# Comments:
# The few-shot result includes examples of how to classify sentiments, which can help the model understand the
# task better and produce more accurate results but in our case. 

# When to choose each:
# - Zero-shot: best for simple, unambiguous tasks where the model already
#   understands the categories well and output format isn't strict - fastest
#   and cheapest, since no example tokens are spent.
# - One-shot: best when you mainly need to lock in a specific OUTPUT FORMAT
#   (e.g., feeding results into a parser downstream) but the classification
#   task itself is easy enough not to need multiple demonstrations.
#   Best when you mainly need to lock in a specific OUTPUT FORMAT (e.g.,
#   feeding results into a parser downstream) but the classification task
#   itself is easy enough not to need multiple demonstrations.
# - Few-shot: best when categories are ambiguous, imbalanced, or easily
#   confused (e.g., distinguishing "mixed" from "negative" for a backhanded
#   compliment), since seeing one example per class anchors the model to a
#   concrete boundary for each. Costs more tokens per call, which matters at
#   scale, but reduces variance on harder or more subjective classification
#   tasks.


# Q4 -- Chain-of-thought reasoning with a labeled final answer
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


# Q5 -- Structured JSON output with sentiment, confidence, and reason

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
    
# Q6 -- Delimiters to separate user text from instructions

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


# Comments:
# Delimiters (triple backticks) clearly mark where the user's input text
# begins and ends, separating it from the surrounding instructions. Without
# them, the model can struggle to tell where the instructions stop and the
# data starts - especially risky if the user's text itself contains words
# that look like instructions. This is also a basic defense against prompt
# injection: content inside the delimited block is treated as DATA to be
# processed, not as commands the model should follow.

# --- Local Models with Ollama ---

# Q1
# Terminal command run (Ollama, local model):
#   ollama run qwen3:0.6b "Explain what a large language model is in two sentences."
#
# Ollama's terminal output (pasted exactly as returned):
# Thinking...
# Okay, the user wants me to explain a large language model in two
# sentences. Let me start by breaking down the key elements. First, a large
# language model is a type of artificial intelligence that can understand
# and generate text. But I need to make sure I cover the main points without
# getting too technical.
#
# I should mention that they have a vast amount of training data and use
# complex algorithms to process information. Then, I can talk about how they
# can create and understand text, like writing stories or answering
# questions. Wait, but the user asked for two sentences. Let me check that
# again. Yes, two sentences. Make sure each sentence is concise and covers
# the essential aspects.
# ...done thinking.
#
# A large language model is a type of artificial intelligence designed to
# understand and generate text, such as language, music, or any form of
# human communication. It leverages massive datasets and advanced algorithms
# to process and comprehend complex information, enabling it to create
# meaningful content and perform tasks like answering questions or writing
# stories.

# Python/OpenAI equivalent - same prompt, run through the API:

prompt = """Explain what a large language model is in two sentences."""

response = get_completion(prompt)
print(f"OpenAI (gpt-4o-mini) response: {response}")

# Comments:
# Ollama (qwen3:0.6b) responds with a more conversational, exploratory style --
# it visibly "thinks out loud" before answering, and the final answer is
# longer and less tightly scoped to "two sentences" than requested.
# OpenAI's response is more concise, technical, and closely follows the
# instruction to keep it to two sentences.
#
# Advantage and disadvantage of running a model locally:
# Running a model locally provides more control over data and privacy, and
# potentially lower latency once the model is loaded, since there's no
# network round-trip to an external API. However, it requires meaningful
# local compute resources and setup/maintenance effort, and a small local
# model like qwen3:0.6b is noticeably less capable at following precise
# instructions (e.g., staying within a strict sentence count) than a larger
# hosted model. Local models also don't automatically receive the latest
# updates/improvements that cloud-based models get pushed regularly.
