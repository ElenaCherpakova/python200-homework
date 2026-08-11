from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator


if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
# --- RAG Concepts --- 
# Concepts Q1

# Comments:
# Scenario A: RAG is a great approach for this scenario. We would store hundreds of PDFs in a document store and 
# retrieve the relevant information when the user asks a question.
# That retrieved content then can be injected into the LLM prompt, allowing the model to answer questions using the most up-to-date
# information from the intenal policy library.

# Scenario B: Fune-tuning is a great approach for this scenario. Since it's required to write product copy in a very specific brand voice,
# fine-tuning can help the model learn and consistently reproduce that style. The 3,000 examples provide enough training
# data for the model to learn patterns that may be difficult to express through prompting alone.

# Scenario C: For this case, prompt engineering is the great approach. Because the user only needs a single short report
# there is no need to fine-tune a model or build RAG system. We can simply include the report in the prompt and instruct the LLM
# to answer questions based on its contents. 


# Concepts Q2
# A confidently wrong answer can be more harmful than saying "Im not sure" because people are more likely to trust and act
# on an answer that sounds certain. Since AI is increasingly used in many areas of sociaty, a hallucination can have serious
# consequences. For example, if an AI assistant confident gives incorrect medical advice, someone might follow it and put their
# health at risk. The tone also matters because a confident and authoritative response makes the information seem more reliable, 
# even when it is incorrect.

# Concepts Q3
# steps = [
# 1. "Extract text from source documents",  
# - Read and extract the relevant text from documents or webpage
# 2. "Split text into chunks" 
# - Break the exctracted text into smaller pieces so they can be searchable
# 3. "Convert text chunks into embeddings" 
# - Convert each text chunk into a numerical vector that represents its meaning
# 4. "Receive the user's query" 
# - The system receives the question or request from the user
# 5. "Embed the user's query" 
# - Convert the user's question into a vector using the same embedding method
# 6. "Retrieve the most relevant chunks" 
# - Compare the query embedding with the document embeddings and find the most relevent chunks
# 7. "Inject retrieved chunks into the prompt" 
# - Add the retrieved information to the prompt so the LLM has relevant source material.
# 8. "Generate a response from the LLM" 
# -  The LLM uses the user's question and the retrieved context to generate an answer.
# ]


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
print(output)

# Comments: 
# Loyalty.txt document was selected because it had the same overlap score (1) as hours and hiring documents
# When multiple documents have the same score, scores.sort (reverse=True) uses
# the document name to break the tie, so loyalty comes before hiring and hours alphabetically in reverse order. 
# This shows a limitation of simple keyword retrieval because the selected document is not actually relevant to the user's question. 

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
# 1. A vector embedding is when text is converted into a set of numbers that represents its meaning. 
# The text can be split into chunksm and each chunk can be converted into a vector and stored into a vector database. 
# so it can later be searching by meaning.
# 2. The chunk with a cosine similarity of 0.85 is more relevant because it is much close to 1 than 0.30.
# A higher similarity score means that the two texts have a more similar meaning or are more closely related in the embedding space.
# 3. Semantic search is more sophisticated because it looks at the overall meaning of the text rather than mathching words one by one.
# This allows it to find relevant chunks even when the exact words from the query do not appear in the chunk, as long as the meaning are similar.

# Q2

# | Feature                    | Keyword RAG                       | Semantic RAG |
# |----------------------------|-----------------------------------|--------------|
# | What is compared?          | Exact word overlap                | Meaning/vector embedding|
# | What is retrieved?         | Full document                     | Relevent chunks|
# | Can it handle synonyms?    | No                                | Yes            |
# | Storage format             | Plain text dictionary             | Vector Database|
# | Relevance score            | Number of overlapping keywords    | Cosine similarity score |

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
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)
        
# Comments:
# Employee benefits:
# The first retrieved chunk is highly relevant because it specifically describes the company benefits program and has a high 
# similarity score of 0.9086. The other retrieved chunks are less relevant because they discuss
# the company's overview and security.

# The model's response sounds confident and specific. It lists many different benefits without using 
# phrases like "Im not sure" or "based on the context"
# An unexpected result is that the security and general overview chunks were also retrieved,
# even though they are not directly related to employee benefits.

# Security policies: 
# The first retrieved chunk is highly relevant because it is specifically 
# about Network and Data Security and has a similarity score of 0.8838. 
# The benefits and general overview chunks are less relevant. 
# The model's response sounds very confident and specific. It provides 
# detailed security information without any uncertainty or hedging. 
# An unexpected result is that the benefits chunk was also retrieved. 
# This shows that semantic retrieval can return chunks that are only 
# somewhat related to the query rather than perfectly relevant.

# Q2

print("------------------similarity_top_k=1---------------------")
query_engine_2 = index.as_query_engine(similarity_top_k=1)

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine_2.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)
        
    query_engine_2 = index.as_query_engine(similarity_top_k=1)

query_engine_3 = index.as_query_engine(similarity_top_k=5)


print("------------------similarity_top_k=5---------------------")
for q in questions:
    print(f"\nQ: {q}")
    response = query_engine_3.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

# Comments:
# With similarity_top_k=1, the model received only the most relevant chunk 
# for each question. The responses were already accurate and focused because 
# the retrieved context was directly related to the question. 
# With similarity_top_k=5, the responses remained mostly correct and became 
# slightly more detailed, but the model also received several less relevant 
# chunks. For example, the security question also retrieved benefits, 
# company overview, partnership, and financial report chunks.
# This shows that more retrieved context is not always better. Additional
# context can provide useful information, but irrelevant chunks can also
# distract the model or increase the chance of an incorrect or hallucinated 
# response. The goal is to retrieve enough relevant context, not simply as 
# much context as possible.

# Q3
print("----------------LlamaIndex Question 3--------------")
query_4 = "What are the biggest challenges BrightLeaf will face in the future?"
response = query_engine.query(query_4)
print("A:", response)
for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)
        
# Comments:
# I expected this query to be difficult because the documents contain
# information about the company's past and current activities, but they do not
# provide reliable information about the company's future.
#
# The model still gave a confident and specific answer about future challenges
# and strategies, even though the retrieved chunks were mostly about the
# company's overview, a 2022 partnership, and financial performance from
# 2021-2025.
#
# This suggests that the model may have generated information that was not
# directly supported by the retrieved context.
#
# To handle this better, I would add a relevance threshold or a check to make
# sure the retrieved context actually supports the answer. The model should
# also be instructed to say that there is not enough information when the
# documents do not contain an answer, instead of making unsupported predictions.

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
# A faithfulness score of 1.0 means the answer is supported by the retrieved
# information. A score of 0.0 means the answer is not supported by the context.
#
# A relevancy score shows how well the answer relates to the question.
# Faithfulness checks if the information is supported by the context, while
# relevancy checks if the answer actually answers the question.
#
# The scores changed between the two queries. The first query had 1.0 for both
# faithfulness and relevancy. The second query had 0.0 for faithfulness and
# 1.0 for relevancy. This happened because the documents had information about
# employee benefits, but did not have information about the company's stock
# price. The answer was relevant to the question, but it was not supported by
# the retrieved context.
#
# LLM-as-a-judge means using another LLM to evaluate the generated answer.
# It is useful for RAG because there can be different correct ways to answer
# the same question, so a simple accuracy metric would not be enough. The LLM
# can check if the answer is relevant and supported by the retrieved context.