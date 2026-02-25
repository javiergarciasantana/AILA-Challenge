import os
import sys
import glob
from typing import List, Dict
from tqdm import tqdm
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURATION ---

# Paths (Updated to your specific setup)
BASE_DIR = "/Users/javiersantana/INF_2024_2025/Trabajo Fin de Grado/AILA-Challenge/archive"
PATH_CASES = os.path.join(BASE_DIR, "Object_casedocs")
PATH_STATUTES = os.path.join(BASE_DIR, "Object_statutes")
PATH_QUERIES = os.path.join(BASE_DIR, "Query_doc.txt")

# Milvus Config
MILVUS_URI = "http://localhost:19530"
DIMENSION = 384
MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking Config (CRITICAL FOR LEGAL DOCS)
CHUNK_SIZE = 500   # Number of words per chunk
OVERLAP = 50       # Overlap to preserve context between chunks

def connect_milvus():
    """Connects to the Milvus Docker Container."""
    print(f"🔌 Connecting to Milvus at {MILVUS_URI}...")
    try:
        client = MilvusClient(uri=MILVUS_URI)
        return client
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        sys.exit(1)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """
    Splits long text into overlapping chunks. 
    Crucial because Embedding models truncate text after ~256-512 tokens.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def load_files(folder_path: str, prefix: str) -> List[Dict]:
    """Loads text files matching a prefix (e.g., 'C' or 'S')."""
    documents = []
    search_pattern = os.path.join(folder_path, f"{prefix}*.txt")
    files = glob.glob(search_pattern)
    
    print(f"📂 Scanning {folder_path}...")
    
    if not files:
        print(f"⚠️  Warning: No files found in {folder_path}.")
        return []

    for filepath in tqdm(files, desc=f"Loading {prefix} Docs"):
        filename = os.path.basename(filepath)
        doc_id = os.path.splitext(filename)[0]
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
                if text:
                    documents.append({
                        "doc_name": doc_id,
                        "text": text
                    })
        except Exception as e:
            print(f"   Error reading {filename}: {e}")
            
    return documents

def load_queries(query_path: str) -> List[Dict]:
    """Loads queries from Query_doc.txt."""
    queries = []
    if not os.path.exists(query_path):
        print(f"⚠️  Query file missing: {query_path}")
        return [{"id": "Q_Dummy", "text": "murder case involving firearm license"}]

    with open(query_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "||" in line:
                parts = line.strip().split("||")
                if len(parts) >= 2:
                    queries.append({"id": parts[0], "text": parts[1]})
    return queries

def index_data(client, collection_name, documents, model):
    """
    Embeds chunks and inserts into Milvus.
    1 Document -> Many Chunks -> Many Vectors
    """
    if not documents:
        return

    # 1. Reset Collection
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    
    client.create_collection(
        collection_name=collection_name,
        dimension=DIMENSION,
        metric_type="COSINE",
        auto_id=True 
    )
    
    print(f"⚙️  Chunking & Indexing {len(documents)} docs into '{collection_name}'...")
    
    all_chunks_data = []

    # 2. Process and Chunk Documents
    for doc in tqdm(documents, desc="Preparing Chunks"):
        # >>> CHUNKING HAPPENS HERE <<<
        text_chunks = chunk_text(doc['text'])
        
        for i, chunk in enumerate(text_chunks):
            all_chunks_data.append({
                "doc_name": doc['doc_name'],  # Link chunk back to original case ID
                "chunk_id": i,                # Order of the chunk
                "text": chunk                 # The actual text content
            })

    # 3. Embed & Insert in Batches
    batch_size = 64
    for i in tqdm(range(0, len(all_chunks_data), batch_size), desc="Embedding & Inserting"):
        batch = all_chunks_data[i : i + batch_size]
        batch_texts = [item['text'] for item in batch]
        
        # Create Vector
        vectors = model.encode(batch_texts)
        
        # Prepare final payload
        insert_payload = []
        for idx, item in enumerate(batch):
            insert_payload.append({
                "vector": vectors[idx],
                "doc_name": item['doc_name'],
                "text_preview": item['text'][:200] # Store snippet for display
            })
            
        client.insert(collection_name=collection_name, data=insert_payload)
        
    print(f"✅ Indexed {len(all_chunks_data)} chunks total.")

def search_collection(client, collection_name, query_vectors, queries):
    """Searches Milvus and aggregates results by Document ID."""
    print(f"\n🔍 Searching {collection_name}...")
    
    results = client.search(
        collection_name=collection_name,
        data=query_vectors,
        limit=5, 
        output_fields=["doc_name", "text_preview"]
    )
    
    # Define the output file
    output_filename = "retrieval_results.txt"
    
    # Use append mode 'a' so that calls for 'cases' and 'statutes' don't overwrite each other
    with open(output_filename, "a", encoding="utf-8") as f:
      print(f"📝 Exporting results for '{collection_name}' to {output_filename}")

      # Process results for each query
      for i, query_hits in enumerate(results):
        query_id = queries[i]['id']
        
        # Aggregate results: We want the best score for each unique document
        doc_scores = {}
        for hit in query_hits:
          doc_name = hit['entity']['doc_name']
          score = hit['distance'] # For COSINE, lower is better (more similar)
          
          # If we see a doc for the first time, or find a chunk with a better score, update it
          if doc_name not in doc_scores or score < doc_scores[doc_name]:
            doc_scores[doc_name] = score

        # Sort the unique documents by their best score
        # The prompt doesn't specify a format, so we'll use a TREC-like format
        # QueryId DocumentId Rank Score RunId
        sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1])

        for rank, (doc_name, score) in enumerate(sorted_docs, 1):
          # Format: QueryId, DocumentId, Rank, Score, RunId
          run_id = f"milvus_{collection_name}"
          f.write(f"{query_id}\t{doc_name}\t{rank}\t{score:.4f}\t{run_id}\n")


def main():
    client = connect_milvus()
    model = SentenceTransformer(MODEL_NAME)
    
    # Load raw text
    cases = load_files(PATH_CASES, "C")
    statutes = load_files(PATH_STATUTES, "S")
    queries = load_queries(PATH_QUERIES)

    # Index (Chunks will be created inside index_data)
    if cases:
        index_data(client, "aila_cases", cases, model)
    if statutes:
        index_data(client, "aila_statutes", statutes, model)

    # Search
    if queries:
        q_vectors = model.encode([q['text'] for q in queries])
        
        if client.has_collection("aila_cases"):
            search_collection(client, "aila_cases", q_vectors, queries)
            
        if client.has_collection("aila_statutes"):
            search_collection(client, "aila_statutes", q_vectors, queries)

if __name__ == "__main__":
    main()