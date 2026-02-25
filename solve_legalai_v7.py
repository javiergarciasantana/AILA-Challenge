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
TREC_OUTPUT_FILE = "test_results/v7/trec_rankings.txt"    # El formato crudo para Kaggle
METRICS_OUTPUT_FILE = "test_results/v7/eval_metrics.txt"  # Tu reporte legible de métricas
RAG_OUTPUT_FILE = "test_results/v7/rag_final_answers.txt"  # New file for LLM answers
MILVUS_URI = "http://localhost:19530"
OLLAMA_BASE_URL = "http://localhost:11434/v1" 
OLLAMA_MODEL = "llama3"

# CUSTOM DENSE MODEL
DENSE_MODEL_PATH = "./models/InLegalBERT-AILA-Tuned" 

# RERANKER
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# PARAMS
TOP_K = 100
FINAL_K = 30
TOP_N_GEN = 3 # Number of docs to pass to LLM for final answer

# METRICS
PATH_QRELS_STATUTES = os.path.join(BASE_DIR, "relevance_judgments_statutes.txt")
PATH_QRELS_CASES = os.path.join(BASE_DIR, "relevance_judgments_priorcases.txt")
K_METRICS = 50

class AgenticLayer:
  """Handles the 'Thinking' part using Ollama"""
  def __init__(self):
    self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
  
  def get_legal_keywords(self, user_query):
    short_query = user_query[:1500] 
    prompt = (
        "You are an expert in Indian Law. Read the facts below and output ONLY a list of "
        "relevant Indian Penal Code (IPC) Sections and specific legal concepts. "
        "DO NOT write introductory sentences. Output ONLY the keywords separated by spaces.\n\n"
        f"FACTS: {short_query}"
    )
    try:
      response = self.client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1 
      )
      legal_terms = response.choices[0].message.content.strip()
      # Limpieza por si acaso
      if ":" in legal_terms: legal_terms = legal_terms.split(":")[-1].strip()
      print(f"🧠 [Ollama Keywords]: {legal_terms}")
      return legal_terms
    except Exception as e:
      print(f"⚠️ Ollama Error: {e}")
      return ""

  def generate_final_answer(self, user_query, context_docs):
    """
    Generates a final answer based on the top retrieved documents.
    """
    context_str = ""
    source_list = []
    
    for i, doc in enumerate(context_docs):
        doc_id = doc['id']
        text_snippet = doc['text'][:2500] # Limit context window (approx 2.5k chars per doc)
        context_str += f"\n--- DOCUMENT ID: {doc_id} ---\n{text_snippet}\n"
        source_list.append(doc_id)
    
    prompt = (
        f"You are a helpful legal assistant for Indian Law. "
        f"Answer the user's query utilizing the information provided in the context below.\n"
        f"Start your answer by explicitly stating: 'Based on the analysis of documents {', '.join(source_list)}...'\n"
        f"If the documents contain relevant statutes or precedents, cite them.\n\n"
        f"USER QUERY: {user_query}\n\n"
        f"LEGAL CONTEXT:\n{context_str}\n\n"
        f"FINAL ANSWER:"
    )

    try:
        response = self.client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip(), source_list
    except Exception as e:
        return f"Error generating answer: {e}", []

class CustomHybridSearch:
  def __init__(self):
    print(f"⏳ Loading Dense Model: {DENSE_MODEL_PATH} (optimized for MPS on Apple Silicon)...")
    self.dense_model = SentenceTransformer(DENSE_MODEL_PATH, device='mps')  # Added MPS for efficiency on Silicon Mac
    self.dense_dim = 768 
    
    print("⏳ Initializing BM25 (Sparse) Model...")
    self.analyzer = build_default_analyzer(language="en")
    self.bm25_ef = BM25EmbeddingFunction(self.analyzer)
    
    print("⏳ Loading Reranker...")
    self.reranker = CrossEncoder(RERANKER_MODEL, device='mps')  # Added MPS for efficiency on Silicon Mac
    
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
          # We need to fit BM25 anyway to run queries later (it needs statistics)
          # We reconstruct a temporary corpus for fitting if cached
          print(f"✅ Collection '{collection_name}' already exists. Re-fitting BM25 for query consistency...")
          temp_chunks = []
          for d in docs[:500]: # Fit on subset to be fast, or full if needed
              temp_chunks.extend(d['text'].split()[:512]) # Dummy chunks
          # Ideally we fit on full corpus, but for speed here we assume user just runs it once properly.
          # For correctness in this script, we will chunk again just for fitting BM25 if needed.
          # BUT: Since self.bm25_ef needs to be fit, we usually do it.
          # For this script, we will just proceed to chunking to ensure BM25 is fit correctly.
          pass 
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
    
    # Check if we really need to insert (after fitting)
    if self.client.has_collection(collection_name) and self.client.query(collection_name, filter="", limit=1):
        print(f"✅ Collection '{collection_name}' ready. Skipping Insertion.")
        return

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

  def search(self, collection_name, dense_query_text, sparse_query_text):
    # 1. Embed Query Dense
    query_dense = self.dense_model.encode([dense_query_text], normalize_embeddings=True)[0]
    
    # 2. Embed Query Sparse (BM25) with FIX
    sparse_matrix = self.bm25_ef.encode_queries([sparse_query_text])
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

def load_qrels(filepaths):
    """Carga y fusiona las respuestas correctas (Ground Truth) desde múltiples archivos."""
    qrels = {}
    for filepath in filepaths:
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        q_id, _, doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
                        if rel > 0: # Solo si es relevante
                            if q_id not in qrels:
                                qrels[q_id] = set()
                            qrels[q_id].add(doc_id)
        except FileNotFoundError:
            print(f"⚠️ Advertencia: No se encontró el archivo {filepath}")
    return qrels

def calculate_metrics(retrieved_docs, relevant_docs, k=50):
    """Calcula Precision@K, Recall@K y Average Precision@K"""
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = [doc for doc in retrieved_k if doc in relevant_docs]
    
    # Precisión: De lo recuperado, ¿cuánto es útil?
    precision = len(relevant_retrieved) / k if k > 0 else 0.0
    
    # Cobertura (Recall): De lo que existe útil, ¿cuánto recuperé?
    recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0.0
    
    # Average Precision (AP): Premia que los relevantes estén al principio
    ap = 0.0
    hits = 0
    for i, doc in enumerate(retrieved_k):
        if doc in relevant_docs:
            hits += 1
            ap += hits / (i + 1)
            
    ap = ap / len(relevant_docs) if relevant_docs else 0.0
    
    return precision, recall, ap

def main():
  # --- 1. Configuración de rutas y métricas ---
  cases = load_files(PATH_CASES, "C")
  statutes = load_files(PATH_STATUTES, "S")
  queries = []
  with open(PATH_QUERIES, errors='ignore') as f:
    for l in f: 
      if "||" in l: queries.append({"id": l.split("||")[0], "text": l.split("||")[1].strip()})

  # --- 2. Cargar Ground Truth (FUSIONANDO AMBOS ARCHIVOS) ---
  print("📂 Cargando juicios de relevancia (Casos y Estatutos)...")
  qrels = load_qrels([PATH_QRELS_STATUTES, PATH_QRELS_CASES])

  agent = AgenticLayer()  # Added agentic layer
  engine = CustomHybridSearch()
  
  engine.process_and_index("aila_cases_v6", cases)
  engine.process_and_index("aila_statutes_v6", statutes)
  
  # --- 3. Limpiar archivos de salida ---
  with open(TREC_OUTPUT_FILE, "w") as f: f.write("")
  with open(METRICS_OUTPUT_FILE, "w") as f: f.write("")
  with open(RAG_OUTPUT_FILE, "w") as f: f.write("")

  print(f"\n🚀 Running Agentic RAG on {len(queries)} queries...")
  
  # --- 4. Variables para promedios ---
  total_precision = 0.0
  total_recall = 0.0
  total_ap = 0.0
  queries_evaluated = 0
  
  for q in tqdm(queries):
    q_id = q['id']
    original_query = q['text']
    
    # A. Reasoning Layer: Extraer SOLO las palabras clave
    legal_keywords = agent.get_legal_keywords(original_query)
    
    # Crear la consulta dispersa (BM25) uniendo el original y las keywords
    sparse_query = f"{original_query} {legal_keywords}"

    # B. Retrieval
    cands_c = engine.search("aila_cases_v6", dense_query_text=original_query, sparse_query_text=sparse_query)
    cands_s = engine.search("aila_statutes_v6", dense_query_text=original_query, sparse_query_text=sparse_query)
    
    # C. Reranking
    ranked_c = engine.rerank_sliding_window(original_query, cands_c, FINAL_K)
    ranked_s = engine.rerank_sliding_window(original_query, cands_s, FINAL_K)
    
    # D. Merge & Save Rankings
    final = ranked_c + ranked_s
    final.sort(key=lambda x: x[1], reverse=True)
    
    # --- ARCHIVO 1: Formato Oficial TREC ---
    with open(TREC_OUTPUT_FILE, "a") as f:
        for rank, (doc_id, score) in enumerate(final):
            f.write(f"{q_id} Q0 {doc_id} {rank+1} {score:.4f} AgenticHybrid\n")
    
    # --- ARCHIVO 2: Cálculo y guardado de Métricas ---
    retrieved_doc_ids = [doc[0] for doc in final]
    
    # AHORA QRELS CONTIENE TANTO LOS CASOS COMO LOS ESTATUTOS CORRECTOS
    if q_id in qrels:
        p, r, ap = calculate_metrics(retrieved_doc_ids, qrels[q_id], k=K_METRICS)
        total_precision += p
        total_recall += r
        total_ap += ap
        queries_evaluated += 1
        
        with open(METRICS_OUTPUT_FILE, "a") as f:
            f.write(f"QUERY: {q_id:<10} | Precision@{K_METRICS}: {p:.4f} | Recall@{K_METRICS}: {r:.4f} | AP@{K_METRICS}: {ap:.4f}\n")
    else:
        with open(METRICS_OUTPUT_FILE, "a") as f:
            f.write(f"QUERY: {q_id:<10} | [Sin evaluación - Falta en Ground Truth]\n")

    # --- E. FINAL GENERATION STEP (NEW) ---
    top_docs = []
    for doc_id, score in final[:TOP_N_GEN]:
        if doc_id in engine.doc_store:
            top_docs.append({'id': doc_id, 'text': engine.doc_store[doc_id]})
    
    answer, sources = agent.generate_final_answer(q['text'], top_docs)
    
    output_str = f"QUERY: {q_id}\nBASED ON: {', '.join(sources)}\nANSWER:\n{answer}\n{'-'*50}\n\n"
    with open(RAG_OUTPUT_FILE, "a") as f:
        f.write(output_str)

  # --- 5. Guardar e imprimir reporte estandarizado al final ---
  if queries_evaluated > 0:
      mean_precision = total_precision / queries_evaluated
      mean_recall = total_recall / queries_evaluated
      map_score = total_ap / queries_evaluated
      
      report = (
          f"\n{'='*50}\n"
          f"📊 REPORTE DE EVALUACIÓN GLOBAL (Casos + Estatutos)\n"
          f"{'='*50}\n"
          f"Consultas evaluadas : {queries_evaluated}\n"
          f"Evaluado en Top-K   : {K_METRICS}\n"
          f"{'-' * 50}\n"
          f"Precision@{K_METRICS}      : {mean_precision:.4f}  (Calidad media)\n"
          f"Recall@{K_METRICS}         : {mean_recall:.4f}  (Cobertura media)\n"
          f"MAP (Mean Avg Prec) : {map_score:.4f}  (Ordenamiento medio)\n"
          f"{'='*50}\n"
      )
      
      print(report)
      with open(METRICS_OUTPUT_FILE, "a") as f:
          f.write(report) 

  print(f"✅ Formato TREC guardado en: {TREC_OUTPUT_FILE}")
  print(f"✅ Métricas guardadas en:    {METRICS_OUTPUT_FILE}")
  print(f"📄 Respuestas RAG en:        {RAG_OUTPUT_FILE}")

if __name__ == "__main__":
  main()