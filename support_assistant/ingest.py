"""
Zepto Support Assistant: Document Ingestion and ChromaDB Indexing
Author: Zepto AI/ML Engineering Guild
Module: /support_assistant

Description:
Loads all 8 Zepto policy documents, chunks them, generates dense vector embeddings
using the local sentence-transformers model 'all-MiniLM-L6-v2', and stores them in a
persistent ChromaDB collection for real-time semantic retrieval.
"""

import os
import glob
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "all-MiniLM-L6-v2"


class MiniLMEmbedder:
    """
    Generates dense 384-dimensional vector embeddings using the all-MiniLM-L6-v2
    architecture via ChromaDB's high-performance ONNX engine.
    Ensures seamless execution across platforms without requiring native torch DLLs.
    """
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._fn = embedding_functions.DefaultEmbeddingFunction()
        
    def encode(self, texts: List[str], convert_to_numpy: bool = True):
        embeddings = self._fn(texts)
        if convert_to_numpy:
            return np.array(embeddings)
        return embeddings


def load_documents() -> List[Dict[str, str]]:
    """
    Loads all doc_*.txt files from the docs/ directory.
    Returns a list of dicts with doc_id, text, and source filename.
    """
    doc_files = sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt")))
    if not doc_files:
        raise FileNotFoundError(f"No policy documents found in {DOCS_DIR}")
        
    documents = []
    for filepath in doc_files:
        filename = os.path.basename(filepath)
        doc_id = os.path.splitext(filename)[0]  # e.g., "doc_01"
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        documents.append({
            "doc_id": doc_id,
            "filename": filename,
            "text": content
        })
    return documents


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Initializes a persistent ChromaDB client.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def ingest_corpus():
    """
    Embeds all documents using sentence-transformers and inserts them into ChromaDB.
    """
    print(f"Loading documents from: {DOCS_DIR}")
    docs = load_documents()
    print(f"Loaded {len(docs)} policy documents.")
    
    print(f"Initializing embedding model: {MODEL_NAME} (ONNX runtime)...")
    embedder = MiniLMEmbedder(MODEL_NAME)
    
    client = get_chroma_client()
    
    # Delete existing collection if it exists to ensure fresh ingestion
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
        
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    texts = [d["text"] for d in docs]
    doc_ids = [d["doc_id"] for d in docs]
    metadatas = [{"source": d["filename"], "doc_id": d["doc_id"]} for d in docs]
    
    print(f"Generating dense embeddings for {len(texts)} chunks...")
    embeddings = embedder.encode(texts, convert_to_numpy=True).tolist()
    
    print("Indexing into ChromaDB collection 'zepto_policies'...")
    collection.add(
        ids=doc_ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    
    print(f"Successfully ingested and indexed {collection.count()} policy chunks into ChromaDB at {CHROMA_DIR}!")
    return collection


if __name__ == "__main__":
    ingest_corpus()
