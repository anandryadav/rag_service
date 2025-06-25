# RAG-enhanced LLM Service

This service enhances the basic LLM connector by implementing a Retrieval-Augmented Generation (RAG) pipeline. It uses a local vector store (FAISS) of SAP data definitions to provide relevant context to the LLM before answering a query.

## Project Structure

```
rag_service/
├── data/
│ └── sap_definitions.csv # Sample SAP data definitions
├── vector_store/
│ ├── faiss_index.bin # FAISS index file
│ └── documents.pkl # Serialized documents for the vector store
├── .env # Environment variables (API Key, Model ID, etc.)
├── indexing.py # Script to build the vector store from SAP definitions
├── main.py # The main FastAPI application code
├── requirements.txt # Python dependencies
└── README.md # This file
```

## Prerequisites

- Python 3.8+
- An OpenAI API Key

## Setup & Configuration

### 1. Clone or Download the Project

Get the source code onto your local machine.

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
   # Navigate to the project directory
   cd rag_service
   
   # Create a virtual environment
   python -m venv venv
   
   # Activate it
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages using pip.

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

The service requires an OpenAI API key to function.

1. Rename the `.env.example` file to `.env` (or create a new `.env` file).
2. Open the `.env` file and add your OpenAI API key and other configurations.:

   ```ini
   # .env
   OPENAI_API_KEY="your_secret_api_key_here"
   OPENAI_URL="https://openrouter.ai/api/v1" # Optional: Change if using a different endpoint
   MODEL_ID="deepseek/deepseek-r1-0528:free" # Example model, replace with your desired model
   ```

4. **Build the Vector Store:**
   Before running the service for the first time, you must run the indexing script. This will read the `data/sap_definitions.csv`, embed the content, and save it to the `vector_store/` directory.
    ```bash
    python indexing.py
    ```

## Running the Service

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
````

### Access the service at `http://localhost:8000/` to interact with the API.

```
    curl --location 'http://localhost:8000/llm/query' \
    --header 'Content-Type: application/json' \
    --data '{"prompt": "What’s the data type and length of MAKT-MAKTX?"}
```

## API Documentation

Once the service is running, you can access the interactive API documentation (provided by Swagger UI) at:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)