import os
import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Configuration
DATA_FILE = "data/sap_definitions.csv"
INDEX_FILE = "vector_store/faiss_index.bin"
DOCS_FILE = "vector_store/documents.pkl"
MODEL_NAME = 'all-MiniLM-L6-v2'

# Create vector_store directory if it doesn't exist
os.makedirs("vector_store", exist_ok=True)

print("Loading data...")
# Load the SAP data definitions
df = pd.read_csv(DATA_FILE)

# Preprocess the data into a list of "documents"
# Each document is a string combining the relevant info for embedding
documents = [
    f"Field Name: {row.field_name}. Description: {row.description}. Data Type: {row.data_type}. Length: {row.length}"
    for index, row in df.iterrows()
]

print(f"Loaded {len(documents)} documents.")
print("Loading sentence transformer model...")
# Load the embedding model
model = SentenceTransformer(MODEL_NAME)

print("Embedding documents...")
# Generate embeddings for each document
embeddings = model.encode(documents, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# Create a FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print(f"Index created with {index.ntotal} vectors.")

# Save the FAISS index
print(f"Saving index to {INDEX_FILE}...")
faiss.write_index(index, INDEX_FILE)

# Save the document list for later retrieval
print(f"Saving documents to {DOCS_FILE}...")
with open(DOCS_FILE, 'wb') as f:
    pickle.dump(documents, f)

print("Indexing complete!")
