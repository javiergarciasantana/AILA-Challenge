"""
AILA-Challenge Agentic Hybrid Retrieval System (v6)
---------------------------------------------------
This program implements an advanced retrieval-augmented generation (RAG) pipeline for legal document search.
It combines dense and sparse retrieval (using InLegalBERT-AILA-Tuned embeddings and BM25-like sparse vectors in Milvus)
with an agentic reasoning layer powered by Ollama (Llama3) to expand queries with legal concepts.
A cross-encoder reranker is used for final ranking.

Key differences from previous version (v5):
- Uses InLegalBERT-AILA-Tuned as the dense encoder for legal-specific embeddings.
- Integrates an agentic layer (Ollama/Llama3) to expand queries with legal reasoning and keywords.
- Improved chunking and indexing for both case law and statutes.
- Refined BM25 sparse vector handling and integration with Milvus hybrid search.
- Checks for existing collections to skip re-indexing if possible.
- Optimized for Apple Silicon Macs using MPS device for GPU acceleration.
- Maintains modular design for easy adaptation and experimentation.
- Added sliding window reranking with focus on key sections (e.g., first and last windows) for better relevance.
"""

import os
import glob
import re
from tqdm import tqdm
import numpy as np
from openai import OpenAI  # For Ollama integration
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, DataType
from sentence_transformers import SentenceTransformer, CrossEncoder

# Import Milvus BM25 Function
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer

# --- CONFIGURATION ---
BASE_DIR = "/Users/javiersantana/INF_2024_2025/Trabajo Fin de Grado/AILA-Challenge/archive"
PATH_CASES = os.path.join(BASE_DIR, "Object_casedocs")
PATH_STATUTES = os.path.join(BASE_DIR, "Object_statutes")
PATH_QUERIES = os.path.join(BASE_DIR, "Query_doc.txt")
OUTPUT_FILE = "test_results/retrieval_results_v6.txt"

MILVUS_URI = "http://localhost:19530"
OLLAMA_BASE_URL = "http://localhost:11434/v1" 
OLLAMA_MODEL = "llama3"

# 1. YOUR CUSTOM DENSE MODEL
DENSE_MODEL_PATH = "./models/InLegalBERT-AILA-Tuned" 

# 2. RERANKER
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# PARAMS
TOP_K = 100
FINAL_K = 30

class AgenticLayer:
  """Handles the 'Thinking' part using Ollama"""
  def __init__(self):
    self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
  
  def expand_query_with_reasoning(self, user_query):
    """
    Translates raw facts into legal concepts.
    Input: "He killed him with a knife."
    Output: "He killed him... LEGAL CONTEXT: Murder, Section 302 IPC, Culpable Homicide."
    """
    try:
      prompt = (
        f"Analyze the following factual query. Extract 3-5 distinct legal keywords, "
        f"relevant statutes (like IPC Sections), and criminal charges that apply in indian law.\n"
        f"Query: \"{user_query}\"\n"
        f"Output ONLY the keywords separated by spaces."
      )
      response = self.client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
      )
      legal_terms = response.choices[0].message.content.strip()
      print("legal_terms", legal_terms)
      # Return combined query: Original Facts + Legal Hallucination
      return f"{user_query} {legal_terms}"
    except Exception as e:
      print(f"⚠️ Ollama Error (Skipping expansion): {e}")
      return user_query

class CustomHybridSearch:
  def __init__(self):
    print(f"⏳ Loading Dense Model: {DENSE_MODEL_PATH} (optimized for MPS on Apple Silicon)...")
    self.dense_model = SentenceTransformer(DENSE_MODEL_PATH, device='mps')  # Added MPS for efficiency on Silicon Mac
    self.dense_dim = 768 
    
    print("⏳ Initializing BM25 (Sparse) Model...")
    self.analyzer = build_default_analyzer(language="en")
    self.bm25_ef = BM25EmbeddingFunction(self.analyzer)
    
    print("⏳ Loading Reranker...")
    self.reranker = CrossEncoder(RERANKER_MODEL, device='mps')  # Added MPS for efficiency
    
    self.client = MilvusClient(uri=MILVUS_URI)
    self.doc_store = {} 

  def fit_bm25(self, corpus_text):
    print(f"📊 Fitting BM25 on {len(corpus_text)} chunks (calculating stats)...")
    self.bm25_ef.fit(corpus_text)

  def create_hybrid_collection(self, name):
    if self.client.has_collection(name):
      self.client.drop_collection(name)
      
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="doc_name", datatype=DataType.VARCHAR, max_length=200)
    
    # 1. Dense Field
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
    
    # 2. Sparse Field
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = self.client.prepare_index_params()
    
    # Index for Dense
    index_params.add_index(
      field_name="dense_vector", 
      index_type="HNSW", 
      metric_type="COSINE", 
      params={"M": 16, "efConstruction": 200}
    )
    
    # Index for Sparse
    index_params.add_index(
      field_name="sparse_vector", 
      index_type="SPARSE_INVERTED_INDEX", 
      metric_type="IP", # Important: Use IP for pre-computed BM25 vectors
      params={"drop_ratio_build": 0.2}
    )

    self.client.create_collection(collection_name=name, schema=schema, index_params=index_params)

  def process_and_index(self, collection_name, docs):
    print(f"\n🚀 Processing {collection_name}...")
    
    # 1. Always load Doc Store
    print(f"📂 Loading text for {collection_name}...")
    for d in docs: self.doc_store[d['doc_name']] = d['text']
    
    # 2. Check Cache (like v5)
    if self.client.has_collection(collection_name):
      try:
        self.client.load_collection(collection_name)
        res = self.client.query(collection_name, filter="", limit=1)
        if res:
          #also here
          self.fit_bm25(all_chunks)
          print(f"✅ Collection '{collection_name}' already exists and is loaded. Skipping Embeddings.")
          return
      except:
        print(f"⚠️ Collection '{collection_name}' seems broken/empty. Re-indexing...")
        self.client.drop_collection(collection_name)

    # 3. Prepare Chunks
    all_chunks = []
    doc_names = []
    
    for doc in tqdm(docs, desc="Chunking"):
      words = doc['text'].split()
      for i in range(0, len(words), 400):
        chunk = " ".join(words[i:i+512])
        all_chunks.append(chunk)
        doc_names.append(doc['doc_name'])

    # 4. Fit BM25
    self.fit_bm25(all_chunks)
    
    # 5. Create Collection
    self.create_hybrid_collection(collection_name)
    
    # 6. Embed and Insert
    batch_size = 50
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexing"):
      batch_text = all_chunks[i:i+batch_size]
      batch_names = doc_names[i:i+batch_size]
      
      # A. Generate Dense
      dense_vecs = self.dense_model.encode(batch_text, normalize_embeddings=True)
      
      # B. Generate Sparse (BM25) with FIX
      sparse_matrix = self.bm25_ef.encode_documents(batch_text)
      
      rows = []
      for j in range(len(batch_text)):
        # --- FIX: Manual Extraction using indptr ---
        start = sparse_matrix.indptr[j]
        end = sparse_matrix.indptr[j+1]
        indices = sparse_matrix.indices[start:end]
        data = sparse_matrix.data[start:end]
        
        sparse_dict = {int(idx): float(val) for idx, val in zip(indices, data)}
        if not sparse_dict:
          # If BM25 ignored all words (e.g. only stop words), add a dummy value.
          # We use index 0 with a negligible weight.
          sparse_dict = {0: 0.00001}
        # -------------------------------------------
        
        rows.append({
          "doc_name": batch_names[j],
          "dense_vector": dense_vecs[j],
          "sparse_vector": sparse_dict,
          "text_preview": batch_text[j][:100]
        })
      
      self.client.insert(collection_name, rows)
      
    print("⏳ Creating Index...")
    self.client.load_collection(collection_name)

  def search(self, collection_name, query_text):
    # 1. Embed Query Dense
    query_dense = self.dense_model.encode([query_text], normalize_embeddings=True)[0]
    
    # 2. Embed Query Sparse (BM25) with FIX
    sparse_matrix = self.bm25_ef.encode_queries([query_text])
    # Extract row 0 manually
    start = sparse_matrix.indptr[0]
    end = sparse_matrix.indptr[1]
    indices = sparse_matrix.indices[start:end]
    data = sparse_matrix.data[start:end]
    query_sparse = {int(idx): float(val) for idx, val in zip(indices, data)}
    
    # 3. Hybrid Search
    req_dense = AnnSearchRequest(
      data=[query_dense],
      anns_field="dense_vector",
      param={"metric_type": "COSINE", "params": {"nprobe": 10}},
      limit=TOP_K
    )
    
    req_sparse = AnnSearchRequest(
      data=[query_sparse],
      anns_field="sparse_vector",
      param={"metric_type": "IP", "params": {"drop_ratio_search": 0.0}},
      limit=TOP_K
    )
    
    res = self.client.hybrid_search(
      collection_name, 
      reqs=[req_dense, req_sparse], 
      ranker=RRFRanker(k=60), 
      limit=TOP_K,
      output_fields=["doc_name"]
    )
    
    # Deduplicate
    seen = set()
    results = []
    for hit in res[0]:
      name = hit['entity']['doc_name']
      if name not in seen:
        results.append(name)
        seen.add(name)
    return results

  def rerank_sliding_window(self, query, doc_ids, top_n):
    """The 'Judge' - Reads full text with sliding window, focusing on key sections for better results"""
    doc_scores = []
    
    for did in doc_ids:
      if did not in self.doc_store: continue
      full_text = self.doc_store[did]
      
      # Create Windows
      words = full_text.split()
      windows = [" ".join(words[i:i+400]) for i in range(0, len(words), 350)]
      
      # Optimize: Check first 3 and last 3 windows (Facts + Verdict) for better relevance
      check_wins = windows[:3] + windows[-3:] if len(windows) > 6 else windows
      if not check_wins: continue
      
      pairs = [[query, w] for w in check_wins]
      scores = self.reranker.predict(pairs)
      best_score = max(scores)
      
      doc_scores.append((did, best_score))
      
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores[:top_n]

def load_files(path, prefix):
  d = []
  for f in glob.glob(os.path.join(path, f"{prefix}*.txt")):
    d.append({"doc_name": os.path.splitext(os.path.basename(f))[0], "text": open(f, errors='ignore').read()})
  return d

def main():
  cases = load_files(PATH_CASES, "C")
  statutes = load_files(PATH_STATUTES, "S")
  queries = []
  with open(PATH_QUERIES, errors='ignore') as f:
    for l in f: 
      if "||" in l: queries.append({"id": l.split("||")[0], "text": l.split("||")[1].strip()})

  agent = AgenticLayer()  # Added agentic layer
  engine = CustomHybridSearch()
  
  engine.process_and_index("aila_cases_v6", cases)
  engine.process_and_index("aila_statutes_v6", statutes)
  
  print(f"\n🚀 Running Agentic RAG on {len(queries)} queries...")
  with open(OUTPUT_FILE, "w") as f:
    for q in tqdm(queries):
      # A. Reasoning Layer (Facts -> Legal Terms)
      enhanced_query = agent.expand_query_with_reasoning(q['text'])
      
      cands_c = engine.search("aila_cases_v6", enhanced_query)
      cands_s = engine.search("aila_statutes_v6", enhanced_query)
      
      ranked_c = engine.rerank_sliding_window(enhanced_query, cands_c, FINAL_K)
      ranked_s = engine.rerank_sliding_window(enhanced_query, cands_s, FINAL_K)
      
      final = ranked_c + ranked_s
      final.sort(key=lambda x: x[1], reverse=True)
      
      for rank, (doc_id, score) in enumerate(final):
        f.write(f"{q['id']} Q0 {doc_id} {rank+1} {score:.4f} AgenticHybrid\n")

if __name__ == "__main__":
  main()