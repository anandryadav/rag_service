import logging
import os
import pickle

import faiss
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from openai import OpenAI, RateLimitError, APITimeoutError
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- Initialization ---
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG-enhanced LLM Service",
    description="Query SAP data definitions using RAG."
)

# Fetch the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_URL")
model_id = os.getenv("MODEL_ID")

# Check if the API key is available
if not api_key:
    logging.critical("OPENAI_API_KEY not found in .env file. Please add it.")
    raise RuntimeError("OPENAI_API_KEY not found in .env file. Please add it.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key, base_url=base_url)

# --- RAG Components Loading ---
try:
    print("Loading RAG components...")
    # Load embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Load FAISS index
    index = faiss.read_index("vector_store/faiss_index.bin")
    # Load the original documents
    with open("vector_store/documents.pkl", 'rb') as f:
        documents = pickle.load(f)
    print("RAG components loaded successfully.")
except FileNotFoundError:
    print("ERROR: RAG components not found. Please run indexing.py first.")
    exit()


# --- Pydantic Models ---
class QueryRequest(BaseModel):
    prompt: str


class QueryResponse(BaseModel):
    response: str
    retrieved_context: str


# --- Helper Function for RAG ---
def retrieve_and_augment(query: str, k: int = 3) -> str:
    """
    Retrieves top-k relevant documents from the vector store and augments the prompt.
    """
    # 1. Embed the query
    query_embedding = embedding_model.encode([query]).astype('float32')

    # 2. Retrieve top-k relevant document indices
    # Latency of this step can be benchmarked here.
    # For a simple script, time.time() before and after would suffice.
    _distances, indices = index.search(query_embedding, k)

    # 3. Fetch the actual documents
    retrieved_docs = [documents[i] for i in indices[0]]
    context = "\n\n".join(retrieved_docs)

    return context


# --- API Endpoint ---
@app.post("/llm/query", response_model=QueryResponse)
async def query_rag_llm(request: QueryRequest):
    """
    Accepts a prompt, retrieves relevant context from SAP data,
    augments the prompt, and returns the LLM's response.
    """
    try:
        # 1. Retrieval
        retrieved_context = retrieve_and_augment(request.prompt)

        # 2. Augmentation
        augmented_prompt = f"""
        Based on the following SAP field definitions, answer the user's question.
        Provide a direct answer and explain using the context.

        --- Context from SAP Data Definitions ---
        {retrieved_context}
        --- End of Context ---

        User Question: {request.prompt}
        """

        # 3. Generation (Call to LLM)
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful SAP assistant."},
                {"role": "user", "content": augmented_prompt}
            ]
        )
        response_text = completion.choices[0].message.content

        return {
            "response": response_text,
            "retrieved_context": retrieved_context
        }

    except RateLimitError:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "OpenAI API rate limit exceeded.")
    except APITimeoutError:
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "OpenAI API request timed out.")
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"An error occurred: {e}")
