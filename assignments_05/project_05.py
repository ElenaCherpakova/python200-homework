from dotenv import load_dotenv
from openai import OpenAI
import json
import ollama


load_dotenv()
client = OpenAI()


TOKEN_BUDGET_THRESHOLD = 2000  # Set a threshold for token usage
total_tokens_used = 0
warned_threshold = set()

# Task 1

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    global total_tokens_used
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    
    if response.usage is not None:
        total_tokens_used += response.usage.total_tokens
    return response.choices[0].message.content


def get_completion_ollama(messages, model="qwen3:0.6b", temperature=0.7):
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature}
    )
    return response["message"]["content"]

def print_token_status():
    """
    Print the running token total and warn if a new threshold was crossed.
    """
    print(f"[Tokens] Running total: {total_tokens_used}")
    current_threshold = total_tokens_used
    if current_threshold >= 1 and current_threshold not in warned_threshold:
        warned_threshold.add(current_threshold)
    print(f"Heads up: you've used over {current_threshold * TOKEN_BUDGET_THRESHOLD} tokens this session.\n")

system_prompt = """
You are a career transition assistant that helps users prepare job application
materials for a new career path. Within a single conversation, you help users:

1. Rewrite resume bullet points to highlight transferable skills relevant to
   their target role/industry
2. Draft a cover letter tailored to the job or industry they're targeting
3. Ask targeted follow-up questions to understand their current skills,
   experience, and career goals before producing polished output

Scope and behavior rules:
- Stay focused on job application materials (resumes, cover letters, and the
  follow-up questions needed to produce them). If the user asks for something
  outside this scope (e.g., general life advice, salary negotiation scripts,
  interview coaching unrelated to written materials), gently redirect them
  back to resume/cover letter work, or note that it's outside what you're
  set up to help with here.
- You do not know the user's specific industry norms, company culture, or
  regional conventions (e.g., resume length, formatting expectations, ATS
  quirks for their field). Say so when relevant, and encourage the user to
  apply their own judgment or research their target industry's norms.
- Every time you produce resume bullet points, a cover letter draft, or any
  other submittable text, remind the user to review and edit it themselves
  before submitting it anywhere. Do not let this reminder get dropped from
  your response, even in a long back-and-forth conversation.
"""

# Comments:
# Deliberate choice: I set the system prompt to be very specific about the assistant's role and scope. 
# This helps ensure that the AI stays focused on the task of helping with job application materials and doesn't drift 
# into unrelated advice. The prompt also includes explicit instructions to remind users to review and edit any generated content, 
# which is important for maintaining quality and relevance in their applications.


# Task 2

def rewrite_bullets_text(bullets: str) -> list[str]:
    bullet_text = "\n".join(f"- {b}" for b in bullets)
    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original. Great to use some metrix to show impact.
    Keep the same number of bullet points, and keep them concise (1-2 lines each).

    Return ONLY a valid JSON list. Each item should have 3 keys:
    "original" (the original bullet), "improved" (your rewritten version) and confidence (float, 0-1, representing
    how confident you are that the improved version is accurate and well-supported by the original text).

    Bullet points:
    ```
    {bullet_text}
    ```
    """ 
    messages = [{"role": "system", "content": system_prompt}, 
                {"role": "user", "content": prompt}]
    response = get_completion(messages, temperature=0)
    

    try:
        result = json.loads(response)
        for item in result:
            confidence = item.get("confidence", 1.0)
            print('Original:', item['original'])
            print('Improved:', item['improved'])
            if confidence < 0.7:
                print(f"Flagged (confidence {confidence:.2f} - review this one carefully).")
                print('Original:', item['original'])
                print('Improved:', item['improved'])

        return result
    except json.JSONDecodeError:
        print("Failed to parse JSON from the response.")
        print("Raw response:", response)
        return []


def rewrite_bullets_file(filepath: str) -> list[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        bullets = [line.strip() for line in f.readlines() if line.strip()]
    if not bullets:
        print(f"No bullet points found in {filepath}")
        return []
    return rewrite_bullets_text(bullets)


# Task 3

def generate_cover_letter(job_title: str, background: str) -> str: 
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """
    
    messages = [{"role": "system", "content": system_prompt}, 
                {"role": "user", "content": prompt}]
    response = get_completion(messages, temperature=0)
    return response.strip()
    
    
job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."

job_title_2 = "Software Developer"
background_2 = "Over seven years of experience as a recruiter and last 3 years focused on the technical roles; recently completed \
a fullstack web development bootcamp and continuing to build my skills through volunteering."

print(generate_cover_letter(job_title, background))
print(generate_cover_letter(job_title_2, background_2))

    
# Comments: 
# I choose particular examples of cover letter openings that are specific, confident, and avoid clichés.
# The few-shot examples help the model understand the desired tone and structure. 
# I also included a clear instruction to match the style and tone of the examples, which helps guide the AI's output.

# Task 4

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    if flagged:
        print(f"Text: {text}")
        print("This message was flagged by our moderation system. "
              "Could you please rephrase your message to be more appropriate?\n")
        return False
    else:
        return True
        

safe_input = "Can you help me write a resume bullet point for my experience as a software engineer?"
print("Safe", is_safe(safe_input))

unsafe_input = "I'm going to publicly humiliate and threaten my hiring manager if I don't get this job."
print("Unsafe", is_safe(unsafe_input))

borderline_input = "I want to write a cover letter that criticizes and insults the hiring manager, but in a funny way."
result = client.moderations.create(model="omni-moderation-latest", input=borderline_input)
print("Borderline results", result.results[0].flagged)
print("Borderline categories:", result.results[0].categories)


# Task 5: 

REVIEW_REMINDER = (
    "\nRemember to review, personalize, and fact-check this before using it "
    "anywhere -- you know your experience and industry better than I do.\n"
)

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    
    filepath = input("if you have a .txt file with resume bullet points, enter the path (or press Enter to skip): ").strip()
    if filepath:
        try:
            result = rewrite_bullets_file(filepath)
            if result:
                print(REVIEW_REMINDER)
                summary = "\n".join(f"- {i['improved']}" for i in result)
                messages.append({"role": "user", "content": f"(User uploaded bullets from {filepath}.)"})
                messages.append({"role": "assistant", "content": f"Rewritten bullets:\n{summary}"})
        except FileNotFoundError:
            print(f"Could not find file: {filepath}. Continuing without it.\n") 
                
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "rewrite" in user_input.lower() and ("bullet" in user_input.lower() or "resume" in user_input.lower()):
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            for item in result:
                print(f"Original: {item['original']}")
                print(f"Improved: {item['improved']}")
            # keep the main conversation aware this happened
            print(REVIEW_REMINDER)
            summary = "\n".join(f"- {i['improved']}" for i in result)
            messages.append({"role": "user", "content": f"(User rewrote resume bullets.) {user_input}"})
            messages.append({"role": "assistant", "content": f"Rewritten bullets:\n{summary}"})

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
            
            cover_letter = generate_cover_letter(job_title, background)
            print("\nJob Application Helper: Here's a draft of your cover letter opening:\n")
            print(cover_letter)
            
            print(REVIEW_REMINDER)
            
            messages.append({"role": "user", "content": f"(User requested a cover letter for {job_title}.) {user_input}"})
            messages.append({"role": "assistant", "content": f"Draft cover letter opening:\n{cover_letter}"})
        # 7. Otherwise, handle it as a regular chat turn
        else:
            # YOUR CODE:
            # - Append the user's message to `messages`
            messages.append({"role": "user", "content": user_input})
            # - Call get_completion(messages)
            # get_completion_response = get_completion(messages)
            get_completion_response = get_completion_ollama(messages)
            # - Print the reply
            print(f"\nJob Application Helper: {get_completion_response}\n")
            # - Append the reply to `messages` as an assistant message
            messages.append({"role": "assistant", "content": get_completion_response})
        print_token_status()

if __name__ == "__main__":
    run_chatbot()
    
    
# Task 6:
# Comments:
# Q1: The bot learned mostly from Western, corporate-style writing, so its advice may favor
# one type of tone or industry over others. For example, it tends to push North American
# resume conventions like adding metrics and numbers to every bullet, but not every industry
# or culture writes resumes that way, so this style might not fit everyone equally well.
#
# Q3: One guardrail I would add is a confidence disclaimer. The bot should always remind
# users that it is an AI and may not have accurate, up-to-date, or industry-specific
# knowledge, and that its output is a starting draft, not a finished product.


# Extra

# Top-p experiment 


def warm_up(prompt, top_p_values=(0.1, 0.5, 1.0)):
    
    messages = [{"role": "user", "content": prompt}]
    
    for p in top_p_values:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages,
            temperature=1,
            max_completion_tokens=200,
            top_p=p
    )
        print('Top p: ', {p})
        print(response.choices[0].message.content)
    

test_prompt="Write a one-sentence tagline for a career coaching app"
warm_up(test_prompt)

# Comments:
# The top_p experiment: 0.1 and 0.5 gave almost identical results, whereas 1.0 gave a
# different result, because it opens access to the full vocabulary, including
# less obvious word choices.
# Unlike temperature, which changes how "bold" the model's word choices are gradually,
# top_p works as a threshold, either a word is accessible or not.
# Therefore, the result changes not smoothly, but in jumps.