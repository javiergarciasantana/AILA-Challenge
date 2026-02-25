from ..auxfunctions import load_objects, load_queries
from isaacus import AsyncClient
import numpy as np
import asyncio
import json
import os

async def generate_embeddings():
    """Generate embeddings for your legal corpus using Isaacus API."""
    # Initialize the async client (API key from environment)
    client = AsyncClient(api_key=os.getenv("ISAACUS_API_KEY"))
    # Your corpus and queries
    
    corpus_texts = load_objects("S", "../archive/Object_statutes")
    queries = load_queries("../archive/Query_doc.txt")
    # Process queries to remove the first part (e.g., "AILA_Q1||")
    processed_queries = [query.split("||", 1)[-1] for query in queries]
    # Generate corpus embeddings with task-aware encoding
    corpus_response = await client.embed(
        model="kanon-2-embedder",
        inputs=corpus_texts,
        task="retrieval/document"  # Tell model these are documents
    )
    corpus_embeddings = np.array(corpus_response.embeddings, dtype=np.float32)
    # Generate sample query embeddings with task-aware encoding
    query_response = await client.embed(
        model="kanon-2-embedder",
        inputs=processed_queries,
        task="retrieval/query"  # Tell model these are queries
    )
    query_embeddings = np.array(query_response.embeddings, dtype=np.float32)
    # Create an embeddings dir if it doesn't exist
    os.makedirs("embeddings", exist_ok=True)
    # Save to disk for later use
    # Prepare data for saving
    corpus_data = {
      f"S{index + 1}": embedding.tolist()
      for index, embedding in enumerate(corpus_embeddings)
    }
    query_data = {
      f"Q_{index + 1}": embedding.tolist()
      for index, embedding in enumerate(query_embeddings)
    }

    # Save to a JSON file
    with open("embeddings/statutes.json", "w") as json_file:
      json.dump(corpus_data, json_file)
    await client.close()

    with open("embeddings/queries.json", "w") as json_file:
      json.dump(query_data, json_file)
    await client.close()
# Run it
asyncio.run(generate_embeddings())