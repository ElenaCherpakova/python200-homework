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
#
# Scenario A:
# RAG is the best approach because the assistant needs to answer questions
# using hundreds of internal PDFs that are updated regularly. RAG can retrieve
# current information from the policy documents without retraining the model
# every time the documents change.
#
# Scenario B:
# Fine-tuning is the best approach because the goal is to consistently
# generate product descriptions in a specific brand voice. The 3,000 examples
# can help the model learn and reproduce the company's writing style.
#
# Scenario C:
# Prompt engineering is the best approach because the user only needs answers
# from one short report. The report can be included directly in the prompt,
# so there is no need to build a RAG system or fine-tune the model.



# Concepts Q2
#
# A confidently wrong answer can be more harmful than saying "I'm not sure"
# because people are more likely to trust and act on an answer that sounds
# certain.
#
# For example, if an AI gives incorrect medical information, a person might
# follow the advice and make an unsafe decision.
#
# Tone affects trust because a confident and authoritative tone can make
# incorrect information sound more reliable, even when the answer is wrong.

# Concepts Q3
#
# steps = [
#     "Extract text from source documents",
#     "Split text into chunks",
#     "Convert text chunks into embeddings",
#     "Receive the user's query",
#     "Embed the user's query",
#     "Retrieve the most relevant chunks",
#     "Inject retrieved chunks into the prompt",
#     "Generate a response from the LLM",
# ]
#
# 1. Extract text from source documents
#    Read and extract the relevant text from the documents or webpages.
#
# 2. Split text into chunks
#    Break the extracted text into smaller pieces so they can be searched.
#
# 3. Convert text chunks into embeddings
#    Convert each text chunk into a numerical vector that represents its meaning.
#
# 4. Receive the user's query
#    The system receives the user's question or request.
#
# 5. Embed the user's query
#    Convert the user's question into a vector using the same embedding method.
#
# 6. Retrieve the most relevant chunks
#    Compare the query embedding with the document embeddings and find the
#    most relevant chunks.
#
# 7. Inject retrieved chunks into the prompt
#    Add the retrieved information to the prompt so the LLM has relevant
#    source material.
#
# 8. Generate a response from the LLM
#    The LLM uses the user's question and the retrieved context to generate
#    an answer.


# Keyword RAG
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
query = "What are your hours on weekends?"
documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}
output = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", output[0][0])

# Comments:
# The selected document is hours.txt because it has the highest overlap score
# of 1. hiring.txt and loyalty.txt also have a score of 1, but hours.txt is
# selected because it is the first highest-scoring match after sorting.

# Keyword Q2
query_2 = "Do you have anything without caffeine?"
output_2 = simple_keyword_retrieval(query_2, documents, verbose=True)
print(output_2)

# Comments: 
# None of the documents were selected because there were no overlapping keywords between
# the query and any of the documents.
# Keyword RAG didn't get this right because menu document contain relevant information about drinks, but 
# it does not contain the exact keyword 'caffeine'. Keyword retrieval only looks for matching words and does not
# understand that the question is related to the drinks in menu.
# I think semantic retrival using embeddings would be better because it can compare the meaning of the query with the 
# meaning of the document, even when the exact keywords do not match. 

# Keyword Q3

# I predict that no relevant document will be selected because none of the words in the query appear in the documents.
# The word like "for" is a stopword so it will be removed from the query before the keyword matching happens.

query_3 = "How do I sign up for rewards?"
output_3 = simple_keyword_retrieval(query_3, documents, verbose=True)
print(output_3)

# Comments:
# My prediction was correct. No relevant content was found because there were no overlapping keywords
# between the query and any of the document. Although loyalty document is clearly related to rewards, the document uses 
# different words such loyalty program and points which keyword retrieval doesnt no recognized as having the same meaning.


# Semantic RAG Concepts

# Q1
# 1. A vector embedding converts text into a set of numbers that represents
#    the meaning of the text. These vectors can be stored and searched later.
#
# 2. The chunk with a cosine similarity of 0.85 is more relevant because 0.85
#    is closer to 1 than 0.30. A higher similarity score means the two texts
#    have more similar meanings.
#
# 3. Semantic search looks at the meaning of the text instead of only matching
#    exact words. This allows it to find relevant information even when the
#    query and document use different words.

# Q2
# | Feature          | Keyword RAG              | Semantic RAG              |
# |------------------|--------------------------|---------------------------|
# | What is compared?| Exact word overlap       | Meaning/vector embeddings |
# | What is retrieved?| Matching documents       | Relevant chunks          |
# | Synonyms?        | Usually no               | Yes                       |
# | Storage format   | Plain text               | Vector index/database     |
# | Relevance score  | Keyword overlap count    | Similarity score          |

# LlamaIndex
docs = SimpleDirectoryReader("../../../python-200-v1/lessons/06_AI_augmentation/resources/brightleaf_pdfs").load_data()
index = VectorStoreIndex.from_documents(docs)
print(type(index._vector_store).__name__)

# Q1
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
        
# Comments:
#
# Employee benefits:
# The first retrieved chunk is highly relevant because it specifically
# describes the company benefits program and has a high similarity score of
# 0.9086. The other retrieved chunks are less relevant because they discuss
# the company's overview and security.
#
# The model's response sounds confident and specific. It lists several
# benefits without showing uncertainty. An unexpected result is that the
# security and general overview chunks were also retrieved, even though they
# are not directly related to employee benefits.
#
# Security policies:
# The first retrieved chunk is highly relevant because it specifically
# discusses Network and Data Security and has a similarity score of 0.8838.
# The benefits and general overview chunks are less relevant to this question.
#
# The model's response also sounds confident and specific. It provides
# detailed security information without showing much uncertainty. An
# unexpected result is that the employee benefits chunk was also retrieved.
# This shows that semantic retrieval can return chunks that are somewhat
# related to the question, but not always perfectly relevant.

# Q2
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


# Comments:
#
# With similarity_top_k=1, the model receives only the most relevant chunk,
# so the response is more focused and uses less context.
#
# With similarity_top_k=5, the model receives more context. This can provide
# additional useful information, but it can also include less relevant chunks.
#
# For this query, both settings produced a useful answer, but the top_k=1
# result was more focused. This shows that retrieving more chunks does not
# always mean getting a better answer.

# Q3
print("----------------LlamaIndex Question 3--------------")

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

# Comments:
# I expected this query to be difficult because the documents contain
# information about BrightLeaf's past and current activities, but they do not
# provide reliable information about the company's future.
#
# The model still gave a confident answer about future challenges even though
# the retrieved information was mostly about the company's overview,
# partnership, and financial performance.
#
# This suggests that the model may generate information that is not directly
# supported by the retrieved context.
#
# To improve the system, I would add a relevance threshold and instruct the
# model to say that there is not enough information when the documents do not
# contain an answer. This would reduce unsupported answers.

# Q4

print("----------------LlamaIndex Question 4--------------")

llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)

# Evaluate faithfulness and relevancy
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

# Comments:
#
# A faithfulness score of 1.0 means the answer is fully supported by the
# retrieved information. A score of 0.0 means the answer is not supported
# by the retrieved context.
#
# A relevancy score measures how well the answer addresses the user's question.
# Faithfulness checks whether the answer is supported by the context, while
# relevancy checks whether the answer is relevant to the question.
#
# The scores changed between the two queries. The first query had 1.0 for both
# faithfulness and relevancy. The second query had 0.0 for faithfulness and
# 1.0 for relevancy. This happened because the documents contained information
# about employee benefits but did not contain information about BrightLeaf's
# current stock price. The response was relevant to the question, but it was
# not supported by the retrieved context.
#
# LLM-as-a-judge means using another LLM to evaluate the generated answer.
# It is useful for RAG because there can be multiple valid ways to answer a
# question, so a simple exact-match accuracy metric would not work well.
# The LLM judge can evaluate whether the answer is relevant and supported
# by the retrieved context.