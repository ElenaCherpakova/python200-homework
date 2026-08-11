from dotenv import load_dotenv
import os
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator


if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# --- RAG Concepts ---

# Concepts Q1
print("Concepts Q1")
print("""
Scenario A: RAG is the best approach because the assistant needs to answer
questions using hundreds of internal PDFs that are updated regularly. RAG can
retrieve current information from the policy documents without retraining the
model every time the documents change.

Scenario B: Fine-tuning is the best approach because the goal is to
consistently generate product descriptions in a specific brand voice. The
3,000 examples can help the model learn and reproduce the company's writing
style.

Scenario C: RAG is the best approach because the assistant only needs to
answer questions about one specific two-page report. Since the report is
small and can be retrieved directly, the document does not need to go
through any general document workflow -- it can simply be provided as
context to the model without training the model on its content.
""")

# Concepts Q2
print("Concepts Q2")
print("""
A confidently wrong answer can be more harmful than saying "I'm not sure"
because people are more likely to trust and act on an answer that sounds
certain.

For example, if an AI assistant incorrectly tells someone that a certain
medication can safely be combined with another medication, the person might
follow that advice and experience a serious health problem.

Tone affects trust because a confident and authoritative tone can make
incorrect information sound reliable. This can cause users to trust and act
on a hallucinated answer instead of checking the information with a reliable
source or qualified professional.
""")

# Concepts Q3
steps = [
    "Generate a response from the LLM",
    "Extract text from source documents",
    "Receive the user's query",
    "Retrieve the most relevant chunks",
    "Convert text chunks into embeddings",
    "Inject retrieved chunks into the prompt",
    "Split text into chunks",
    "Embed the user's query",
]

arranged_steps = [
    "Extract text from source documents",
    "Split text into chunks",
    "Convert text chunks into embeddings",
    "Receive the user's query",
    "Embed the user's query",
    "Retrieve the most relevant chunks",
    "Inject retrieved chunks into the prompt",
    "Generate a response from the LLM",
]

print("Concepts Q3")
print("Original list:")
for step in steps:
    print(f"- {step}")

print("\nCorrect RAG pipeline order:")
for i, step in enumerate(arranged_steps, start=1):
    print(f"{i}. {step}")

print("""
1. Extract text from source documents
   Read and extract the relevant text from the documents or webpages.

2. Split text into chunks
   Break the extracted text into smaller pieces so they can be searched
   efficiently.

3. Convert text chunks into embeddings
   Convert each text chunk into a numerical vector that represents its
   meaning.

4. Receive the user's query
   The system receives the user's question or request.

5. Embed the user's query
   Convert the user's question into a vector using the same embedding
   method.

6. Retrieve the most relevant chunks
   Compare the query embedding with the document embeddings and find the
   most relevant chunks.

7. Inject retrieved chunks into the prompt
   Add the retrieved chunks to the prompt so the LLM has relevant source
   material.

8. Generate a response from the LLM
   The LLM uses the user's question and retrieved context to generate an
   answer.
""")

# --- Keyword RAG ---
import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# Keyword Q1
print("\nKeyword Q1")
query = "What are your hours on weekends?"
documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}
output = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", output[0][0])

print("""
Comment: hours.txt, hiring.txt, and loyalty.txt all tie with an overlap score
of 1. hours.txt is selected because it is the first best match among the tied
scores after sorting -- not because it has a uniquely higher score.
""")

# Keyword Q2
print("Keyword Q2")
query_2 = "Do you have anything without caffeine?"
output_2 = simple_keyword_retrieval(query_2, documents, verbose=True)
print(output_2)

print("""
Comment: The function returned ("None found", "No relevant content."). No
document was selected because none of the filtered query words overlapped
with any document's text.

This shows a limitation of keyword retrieval. The menu document contains
drinks that could be relevant to a question about caffeine, but it does not
contain the exact word "caffeine". Keyword retrieval only matches literal
words and does not understand that some drinks may be caffeine-free or
related to the concept of caffeine.

Semantic retrieval would work better because embeddings compare the meaning
of the query with the meaning of the document, even when the exact keywords
are different.
""")

# Keyword Q3
print("Keyword Q3")
query_3 = "How do I sign up for rewards?"
output_3 = simple_keyword_retrieval(query_3, documents, verbose=True)
print(output_3)

print("""
Prediction: I expected this query to be difficult for keyword retrieval
because the loyalty document discusses a "loyalty program" and "points",
while the query uses the word "rewards". Keyword retrieval does not
understand that these terms can have similar meanings.

Was the prediction correct? Yes -- the function did not select loyalty.txt,
confirming the prediction. This was not surprising, since it follows directly
from how the function only matches literal overlapping words: "rewards" never
appears in loyalty.txt, so there was no token overlap to score on, even
though the document is clearly the right answer conceptually.
""")

# --- Semantic RAG Concepts ---

# Semantic Q1
print("Semantic Q1")
print("""
1. A vector embedding converts text into a set of numbers that represents
   the meaning of the text. These vectors can be stored and searched later.

2. The chunk with a cosine similarity of 0.85 is more relevant because 0.85
   is closer to 1 than 0.30. A higher similarity score means the two texts
   have more similar meanings.

3. Semantic search looks at the meaning of the text instead of only matching
   exact words. This allows it to find relevant information even when the
   query and document use different words.
""")

# Semantic Q2
print("Semantic Q2")
print("""
| Feature            | Keyword RAG            | Semantic RAG               |
|---------------------|------------------------|-----------------------------|
| What is compared?   | Exact word overlap     | Meaning/vector embeddings   |
| What is retrieved?  | Matching documents     | Relevant chunks             |
| Synonyms?           | Usually no             | Yes                         |
| Storage format      | Plain text             | Vector index/database       |
| Relevance score     | Keyword overlap count  | Similarity score            |
""")

# --- LlamaIndex ---

# LlamaIndex Q1
print("LlamaIndex Q1")
docs = SimpleDirectoryReader("../../../python-200-v1/lessons/06_AI_augmentation/resources/brightleaf_pdfs").load_data()
index = VectorStoreIndex.from_documents(docs)
print(type(index._vector_store).__name__)
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)

    print("Top 3 Retrieved Sources:")
    for i, node_with_score in enumerate(response.source_nodes[:3], start=1):
        print(f"Source {i}:")
        print(f"Document: {node_with_score.node.metadata['file_name']}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(
            f"Text Snippet: "
            f"{node_with_score.node.get_content()[:150]}..."
        )
        print("-" * 30)

print("""
Comment -- Employee benefits:
The query retrieves exactly three source nodes. The first source has the
highest similarity score and is the most relevant because it specifically
discusses BrightLeaf's employee benefits. The other two sources are less
relevant because they contain general company or security information.

The response sounds confident and specific. It provides several benefits
without expressing uncertainty. The retrieval of unrelated security and
company overview chunks shows that semantic retrieval can include partially
related information.

Comment -- Security policies:
The query also retrieves exactly three source nodes. The first source has
the highest similarity score and is highly relevant because it discusses
BrightLeaf's Network and Data Security policies. The other retrieved chunks
are less directly related.

The response sounds confident and specific and provides detailed security
information. An unexpected result is that an employee benefits chunk is also
retrieved, showing that the top three results are not necessarily all
directly relevant.
""")

# LlamaIndex Q2
print("LlamaIndex Q2")
q = "What employee benefits does BrightLeaf offer?"

query_engine_1 = index.as_query_engine(similarity_top_k=1)
query_engine_5 = index.as_query_engine(similarity_top_k=5)

print("------------------similarity_top_k=1---------------------")

response_1 = query_engine_1.query(q)

print(f"Q: {q}")
print("A:", response_1)

for i, node_with_score in enumerate(response_1.source_nodes, start=1):
    print(f"Source {i}:")
    print(f"Document: {node_with_score.node.metadata['file_name']}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:150]}")
    print("-" * 30)

print("------------------similarity_top_k=5---------------------")

response_5 = query_engine_5.query(q)

print(f"Q: {q}")
print("A:", response_5)

for i, node_with_score in enumerate(response_5.source_nodes, start=1):
    print(f"Source {i}:")
    print(f"Document: {node_with_score.node.metadata['file_name']}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:150]}")
    print("-" * 30)

print(f"""
Comment:
With similarity_top_k=1, the model receives only the most relevant chunk,
so the response is more focused and uses less context.

With similarity_top_k=5, the model receives more context. This can provide
additional useful information, but it can also include less relevant chunks.

Comparing the two responses directly: response_1 was --
"{response_1}"
while response_5 was --
"{response_5}"
For this query, both settings produced a useful answer, but the top_k=1
result was more narrowly focused on benefits alone, while the top_k=5
response pulled in more surrounding detail. This shows that retrieving more
chunks does not always change or improve the answer -- it mainly changes how
much supporting context is available.
""")

# LlamaIndex Q3
print("LlamaIndex Q3")
query_4 = "What are the biggest challenges BrightLeaf will face in the future?"
response = query_engine.query(query_4)

print("Q:", query_4)
print("A:", response)

print("Retrieved Sources:")

for i, node_with_score in enumerate(response.source_nodes, start=1):
    print(f"Source {i}:")
    print(f"Document: {node_with_score.node.metadata['file_name']}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
    print("-" * 30)

print("""
Comment:
I expected this query to be difficult because the documents contain
information about BrightLeaf's past and current activities, but they do not
provide reliable information about the company's future.

The model still gave a confident answer about future challenges even though
the retrieved information was mostly about the company's overview,
partnership, and financial performance.

This suggests that the model may generate information that is not directly
supported by the retrieved context.

To improve the system, I would add a relevance threshold and instruct the
model to say that there is not enough information when the documents do not
contain an answer. This would reduce unsupported answers.
""")

# LlamaIndex Q4
print("LlamaIndex Q4")

llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)

faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

q_2 = "What is BrightLeaf's stock price today?"
response_2 = query_engine.query(q_2)
faithfulness_result_2 = faithfulness_evaluator.evaluate_response(query=q_2, response=response_2)
print("Faithfulness Evaluation 2: " + str(faithfulness_result_2.score))

relevancy_result_2 = relevancy_evaluator.evaluate_response(query=q_2, response=response_2)
print("Relevancy Result 2: " + str(relevancy_result_2.score))

print("""
Comment:
Faithfulness measures whether the answer is supported by the retrieved
context. A score of 1.0 means the response is fully grounded in the
retrieved information, while a score of 0.0 means the response contains
information that is not supported by the context at all.

Relevancy measures whether the response is related to the user's question,
independent of whether it's factually grounded. Faithfulness and relevancy
therefore measure different things: an answer can be relevant to a question
but still contain information that is not supported by the retrieved
documents.

For the employee benefits question, both scores were 1.0. This indicates
that the answer was relevant to the question and fully supported by the
retrieved BrightLeaf documents.

For the stock price question, faithfulness dropped to 0.0 because the
documents did not contain BrightLeaf's current stock price -- the model's
answer wasn't grounded in any retrieved content. The response was still
evaluated as relevant because it addressed the stock-price question topic,
even though the information itself was not supported by the retrieved
context.

This demonstrates why both evaluators are useful together: relevancy alone
does not prove that an answer is factually supported by the retrieved
documents.

LLM-as-a-judge means using another LLM to evaluate a generated answer. It is
useful for RAG because there can be multiple valid ways to phrase a correct
answer, making simple exact-match accuracy insufficient. The evaluator LLM
can instead judge whether the response is relevant to the query and grounded
in the retrieved context, which a rigid string comparison could not do.
""")