import os
import sys
import glob
import re
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# --- 1. CONFIGURATION ---

# Update paths to your exact locations
BASE_DIR = "/Users/javiersantana/INF_2024_2025/Trabajo Fin de Grado/AILA-Challenge/archive"
PATH_CASES = os.path.join(BASE_DIR, "Object_casedocs")
PATH_STATUTES = os.path.join(BASE_DIR, "Object_statutes")
PATH_QUERIES = os.path.join(BASE_DIR, "Query_doc.txt")
OUTPUT_FILE = "retrieval_results_optimized.txt"

# Milvus Config
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "aila_hybrid_v1"
DIMENSION = 768  # BGE-base uses 768 dimensions (vs 384 for MiniLM)

# Models
# Bi-Encoder: Fast, good for initial retrieval
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5" 
# Cross-Encoder: Slow but very accurate, used for re-ranking top candidates
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Strategy Params
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 50  # Get top 50 candidates from Vector + Keyword
TOP_N_RERANK = 10     # Return top 10 after re-ranking

# --- 2. HELPER CLASSES ---

class TextProcessor:
  @staticmethod
  def clean(text):
    # Basic cleanup to help BM25
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

  @staticmethod
  def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= chunk_size:
      return [text]
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
      chunk = " ".join(words[i:i + chunk_size])
      chunks.append(chunk)
    return chunks

class HybridSearchEngine:
  def __init__(self):
    print(f"⏳ Loading models...")
    self.bi_encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    self.cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)
    self.bm25 = None
    self.bm25_corpus_map = {} # Maps BM25 index to (doc_name, text_chunk)
    
    # Connect to Milvus
    try:
      self.milvus = MilvusClient(uri=MILVUS_URI)
      print("✅ Connected to Milvus")
    except Exception as e:
      print(f"❌ Milvus Connection Failed: {e}")
      sys.exit(1)

  def index_data(self, documents: List[Dict]):
    """
    1. Indexes data into Milvus (Dense) if collection doesn't exist.
    2. Builds BM25 index in memory (Sparse) on every run.
    """
    print(f"⚙️  Processing {len(documents)} documents...")
    
    all_chunks = []
    bm25_tokens = []
    
    # Prepare Chunks for BM25 and potentially Milvus
    for doc in tqdm(documents, desc="Chunking"):
      chunks = TextProcessor.chunk_text(doc['text'])
      for chunk in chunks:
        all_chunks.append({
          "doc_name": doc['doc_name'],
          "text": chunk
        })
        bm25_tokens.append(TextProcessor.clean(chunk).split())
    
    # Always build the in-memory BM25 index
    print("📊 Building BM25 Index...")
    self.bm25 = BM25Okapi(bm25_tokens)
    self.bm25_corpus_map = {i: chunk_data for i, chunk_data in enumerate(all_chunks)}

    # Check if Milvus collection exists and is populated
    if self.milvus.has_collection(COLLECTION_NAME):
      stats = self.milvus.get_collection_stats(collection_name=COLLECTION_NAME)
      if stats['row_count'] > 0:
        print(f"✅ Collection '{COLLECTION_NAME}' already exists with {stats['row_count']} entities. Skipping Milvus indexing.")
        return # Skip Milvus indexing part

    # If collection doesn't exist or is empty, create and populate it.
    if self.milvus.has_collection(COLLECTION_NAME):
      self.milvus.drop_collection(COLLECTION_NAME) # Drop if empty
    
    self.milvus.create_collection(
      collection_name=COLLECTION_NAME,
      dimension=DIMENSION,
      metric_type="COSINE",
      auto_id=True
    )
    
    print("🧠 Generating Vectors & Indexing to Milvus...")
    batch_size = 64
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Milvus Insert"):
      batch = all_chunks[i : i + batch_size]
      texts = [x['text'] for x in batch]
      vectors = self.bi_encoder.encode(texts, normalize_embeddings=True)
      
      data = []
      for idx, vec in enumerate(vectors):
        data.append({
          "vector": vec,
          "doc_name": batch[idx]['doc_name'],
          "text": batch[idx]['text'][:300] # Store snippet
        })
      self.milvus.insert(collection_name=COLLECTION_NAME, data=data)
      
    print(f"✅ Indexed {len(all_chunks)} chunks into Milvus successfully.")

  def search(self, query_text: str):
    """
    Performs Hybrid Search + Reranking
    """
    # A. BM25 Search (Keyword)
    tokenized_query = TextProcessor.clean(query_text).split()
    
    candidates = {} # doc_name -> {score, text}

    # Collect BM25 Candidates
    bm25_scores = self.bm25.get_scores(tokenized_query)
    top_indices = np.argsort(bm25_scores)[::-1][:TOP_K_RETRIEVAL]
    
    for idx in top_indices:
      chunk_data = self.bm25_corpus_map[idx]
      doc_name = chunk_data['doc_name']
      # We treat the text as the candidate
      candidates[doc_name] = chunk_data['text']

    # B. Milvus Search (Semantic)
    query_vector = self.bi_encoder.encode([query_text], normalize_embeddings=True)
    milvus_res = self.milvus.search(
      collection_name=COLLECTION_NAME,
      data=query_vector,
      limit=TOP_K_RETRIEVAL,
      output_fields=["doc_name", "text"]
    )
    
    for hit in milvus_res[0]:
      doc_name = hit['entity']['doc_name']
      if doc_name not in candidates:
        candidates[doc_name] = hit['entity']['text']

    # C. Re-Ranking (The Magic Step)
    pairs = []
    candidate_ids = list(candidates.keys())
    
    for doc_id in candidate_ids:
      pairs.append([query_text, candidates[doc_id]])
    
    if not pairs:
      return []

    # Predict scores
    rerank_scores = self.cross_encoder.predict(pairs)
    
    # Sort by Re-ranker score
    results = []
    for i, score in enumerate(rerank_scores):
      results.append((candidate_ids[i], score))
    
    # Sort descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:TOP_N_RERANK]

# --- 3. MAIN LOGIC ---

def load_files(folder_path, prefix):
  docs = []
  pattern = os.path.join(folder_path, f"{prefix}*.txt")
  files = glob.glob(pattern)
  print(f"📂 Loading {len(files)} from {folder_path}...")
  for fpath in files:
    name = os.path.splitext(os.path.basename(fpath))[0]
    with open(fpath, 'r', errors='ignore') as f:
      docs.append({"doc_name": name, "text": f.read()})
  return docs

def load_queries(path):
  queries = []
  with open(path, 'r', errors='ignore') as f:
    for line in f:
      if "||" in line:
        p = line.strip().split("||")
        queries.append({"id": p[0], "text": p[1]})
  return queries

def main():
  engine = HybridSearchEngine()

  # Load Data
  cases = load_files(PATH_CASES, "C")
  statutes = load_files(PATH_STATUTES, "S")
  all_docs = cases + statutes
  
  # Index
  engine.index_data(all_docs)

  # Load Queries
  queries = load_queries(PATH_QUERIES)
  
  # Search & Write Results
  print(f"\n🚀 Running Search for {len(queries)} queries...")
  
  with open(OUTPUT_FILE, "w") as f_out:
    for q in tqdm(queries, desc="Searching"):
      short_query = q['text'][:1000] 
      
      results = engine.search(short_query)
      
      # Writing standard TREC format (QID DOCID RANK SCORE TAG)
      for rank, (doc_id, score) in enumerate(results):
        line = f"{q['id']} {doc_id} {rank+1} {score:.4f} Hybrid_Rerank\n"
        f_out.write(line)

  print(f"\n✅ Results saved to {OUTPUT_FILE}. Run your evaluator on this file.")
  print("   (Note: You might need to adapt the output format in 'main' to match exactly what your check_results.py expects)")

if __name__ == "__main__":
  main()
