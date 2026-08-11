from dotenv import load_dotenv
from pathlib import Path

from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Step 1: Setup
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


docs_dir = Path("../../../python-200-v1/lessons/06_AI_augmentation/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# Step 2: Load Documents
print('--------- Step 2: Load the Documents------------')
docs = SimpleDirectoryReader("../../../python-200-v1/lessons/06_AI_augmentation/resources/groundwork_docs").load_data()
print(f"Number of documents loaded: {len(docs)}\n")

print(f"The file name of each document:")
for i, doc in enumerate(docs, start=1):
    print(f"Document {i}: {doc.metadata.get('file_name', 'Unknown')}")

print("-" * 40)


# Step 3: Build the Index
print("--------Step 3: Build the Index--------")

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)

print("Vector index created successfully.")
print(f"Number of documents indexed: {len(docs)}")
print("Indexed files:")

for i, doc in enumerate(docs, start=1):
    print(f"  {i}. {doc.metadata.get('file_name', 'Unknown')}")

print("-" * 40)

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
    print("Top Retrieved Source Node:")
    print(f"Document name: {top_node.node.metadata['file_name']}")
    print(f"Similarity Score: {top_node.score:.4f}")
    print(f"Text Snippet (first 200 characters): {top_node.node.get_content()[:200]}...")
    print("-" * 30)
        
        
# Comments:
# The assistant sounded confident and the answers were mostly accurate and
# relevant for all five questions. The answers about the weekend hours, loyalty
# program, company history, and catering/wholesale were clearly supported by
# the retrieved documents.
#
# The dairy-free milk answer was a little surprising because the retrieved
# document was seasonal_specials.txt instead of the main menu document. After
# increasing the number of retrieved results, the relevant dairy-free
# information appeared in the retrieved context.
#
# Overall, the results show that the assistant can give confident and useful
# answers when the relevant information is available, but the retrieved source
# should still be checked to make sure the answer is actually supported.

# Step 5: Find a Failure
print('--------Step 5: Find a Failure---------')

q = "What is Groundwork Coffee's revenue for 2025?"
response = query_engine.query(q)
print("Q:", q)
print("A:", response)

print("All Retrieved Source Nodes:")

for i, node_with_score in enumerate(response.source_nodes, start=1):
    print(f"\nSource Node {i}:")
    print(f"Document: {node_with_score.node.metadata['file_name']}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
    print("-" * 30)

# Comments:
#
# What I asked and why:
# I asked for Groundwork Coffee's revenue for 2025 because I did not expect
# this information to be in the provided documents. I expected the assistant
# to struggle because the answer was not available in the context.
#
# What went wrong:
# The retrieval returned chunks from the documents, but they did not contain
# information about Groundwork Coffee's 2025 revenue. The model did not guess
# a revenue number and instead said that the information was not available.
# This means the main problem was missing information rather than the model
# inventing an answer.
#
# Model tone:
# The model became less confident and clearly stated that it could not provide
# the answer from the available context. This is a good behavior because it
# avoided hallucinating a number. It also shows that AI-generated answers
# should still be checked against the retrieved sources instead of being
# trusted automatically.
#
# How I would improve the system:
# I would add a relevance threshold to detect when the retrieved chunks are
# not relevant enough to answer the question. I would also instruct the model
# to say that the information is not available when the documents do not
# contain an answer.


# Step 6: Reflection

# Comments:
# 1. The manual implementation took about 100 lines, while the LlamaIndex
# implementation took about 20 lines. This shows how a framework can greatly
# reduce the amount of code needed and let us focus more on the application
# instead of implementing the RAG pipeline ourselves.

# 2. One useful use case would be an internal HR assistant. Employees could
# ask questions about company policies, benefits, vacation rules, or other
# information stored in internal documents. This would make it easier for
# employees to find information without searching through many documents.

# 3. One failure mode RAG cannot fully prevent is when the retrieved documents
# contain incorrect or outdated information. Even if the retrieval works
# correctly, the model can still generate an answer based on that incorrect
# information.