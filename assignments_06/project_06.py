from dotenv import load_dotenv
from pathlib import Path

from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

# Step 1: Setup
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
docs_dir = Path("../../../python-200-v1/lessons/06_AI_augmentation/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# Step 2: Load the Documents
print('--------- Step 2: Load the Documents------------')
docs = SimpleDirectoryReader("../../../python-200-v1/lessons/06_AI_augmentation/resources/groundwork_docs").load_data()
print(f"Number of documents loaded: {len(docs)}\n")

print(f"The file name of each document:")
for doc in docs:
    print(f"File name: doc.metadata['file_name']")


# Step 3: Build the Index and Query Engine
print('----Step 3: Build the Index and Query Engine----')
index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine(similarity_top_k=3)
print('Index built successfully. Ready to answer questions.\n')

# Step 4: Query the Assistant
print('----Step 4: Query the Assistant----')

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)

    top_node = response.source_nodes[0]
    print(f"Document name: {top_node.node.metadata['file_name']}")
    print(f"Similarity Score: {top_node.score:.4f}")
    print(f"Text Snippet (first 200 characters): {top_node.node.get_content()[:200]}...")
    print("-" * 30)
        
        
# Comments:
# The assistant sounded confident and the answers were mostly accurate and
# relevant to the questions. The retrieved documents also matched the topics
# well. The answer about the loyalty program was especially detailed, and the
# answer about how Groundwork started was supported by the our_story.txt file.
# Nothing was too surprising because the answers were based on the provided
# documents.

# Step 5: Find a Failure
print('--------Step 5: Find a Failure---------')

q = "What is Groundwork Coffee's revenue for 2025?"
response = query_engine.query(q)
print("Q:", q)
print("A:", response)
top_node = response.source_nodes[0]
print(f"Document: {top_node.node.metadata['file_name']}")
print(f"Similarity Score: {top_node.score:.4f}")
print(f"Text Snippet: {top_node.node.get_content()[:200]}...")
print("-" * 30)

#Comments:
# I asked the model to provide the company's revenue for 2025 because this 
# information was not expected to be in the provided documents, so I expected 
# the assistant to struggle with the question.
# The retrieval didnt find information about the company's revenue and 
# model didnt guess an answer, instead, it corretly said that the information
# wasn't available in the provided context.
# The model also became more cautions and clearly said that it could not 
# provide the answer. It's actually a good example of the system avoiding hallucination.
# This proof that AI answers should still be checked
# against the retrieved information instead of being trusted automatically.
# To improve the system, I would add better handling for questions that are
# outside the available documents and make the assistant clearly say when the information is not available.


# Step 6: Reflection
print('--------Step 6: Reflection--------')

# Comments:
# 1. The LlamaIndex implementation took much fewer lines of code compared to
# doing the chunking, embedding, and indexing manually. This shows that a
# framework like LlamaIndex can save time and make it easier to build a RAG
# system without writing all of the underlying code ourselves.

# 2. One useful use case would be an internal HR assistant. Employees could
# ask questions about company policies, benefits, vacation rules, or other
# information stored in internal documents. This would make it easier for
# employees to find information without searching through many documents.

# 3. One failure mode RAG cannot fully prevent is when the retrieved documents
# contain incorrect or outdated information. Even if the retrieval works
# correctly, the model can still generate an answer based on that incorrect
# information.