"""
AILA-Challenge Agentic Hybrid Retrieval System (v8 - Asymmetric RAG)
---------------------------------------------------
This program implements the FINAL retrieval-augmented generation (RAG) pipeline.
It combines dense retrieval (InLegalBERT) and sparse retrieval (BM25) in Milvus.

KEY INNOVATION (Asymmetric Querying):
- The Agent (Llama 3) expands the query with deduced legal keywords.
- DENSE ENGINE (InLegalBERT) receives the EXPANDED query to understand deep context.
- SPARSE ENGINE (BM25) receives ONLY the ORIGINAL query to prevent exact-match noise and MAP degradation.
- Cross-encoder rerankers have been removed following ablation studies demonstrating Domain Shift degradation.
"""

import os
import glob
import re
from tqdm import tqdm
import unicodedata
from openai import OpenAI  
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, DataType
from sentence_transformers import SentenceTransformer

from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer

# --- CONFIGURATION ---
BASE_DIR = "/Users/javiersantana/INF_2024_2025/Trabajo Fin de Grado/AILA-Challenge/archive"
PATH_CASES = os.path.join(BASE_DIR, "Object_casedocs")
PATH_STATUTES = os.path.join(BASE_DIR, "Object_statutes")
PATH_QUERIES = os.path.join(BASE_DIR, "Query_doc.txt")
MILVUS_URI = "http://localhost:19530"

# OLLAMA
OLLAMA_BASE_URL = "http://localhost:11434/v1" 
OLLAMA_MODEL = "llama3"

# CUSTOM DENSE MODEL
DENSE_MODEL_PATH = "./models/InLegalBERT-AILA-Tuned" 

# PARAMS
CHUNK_SIZE = 300
OVERLAP = 50
TOP_K = 100
FINAL_K = 30
TOP_N_GEN = 3 

# METRICS
K_METRICS = 60
PATH_QRELS_STATUTES = os.path.join(BASE_DIR, "relevance_judgments_statutes.txt")
PATH_QRELS_CASES = os.path.join(BASE_DIR, "relevance_judgments_priorcases.txt")

os.makedirs("test_results/v8", exist_ok=True)
TREC_OUTPUT_FILE = "test_results/v8/trec_rankings.txt"    
METRICS_OUTPUT_FILE = "test_results/v8/eval_metrics.txt"  
RAG_OUTPUT_FILE = "test_results/v8/rag_final_answers.txt"  

class TextProcessor:
    @staticmethod
    def super_clean(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def chunk_text(text):
        words = text.split()
        if not words: return []
        chunks = []
        for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
            chunk = " ".join(words[i:i + CHUNK_SIZE])
            chunks.append(chunk)
        return chunks

class AgenticLayer:
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
      if ":" in legal_terms: legal_terms = legal_terms.split(":")[-1].strip()
      print(f"🧠 [Ollama Keywords]: {legal_terms}")
      return legal_terms
    except Exception as e:
      print(f"⚠️ Ollama Error: {e}")
      return ""

  def generate_final_answer(self, user_query, context_docs):
    context_str = ""
    source_list = []
    
    for i, doc in enumerate(context_docs):
        doc_id = doc['id']
        text_snippet = doc['text'][:2500] 
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
    print(f"⏳ Loading Dense Model: {DENSE_MODEL_PATH}...")
    self.dense_model = SentenceTransformer(DENSE_MODEL_PATH, device='mps')
    self.dense_dim = 768 
    
    print("⏳ Initializing BM25 (Sparse) Model...")
    self.analyzer = build_default_analyzer(language="en")
    self.bm25_ef = BM25EmbeddingFunction(self.analyzer)
    
    self.client = MilvusClient(uri=MILVUS_URI)
    self.doc_store = {} 

  def fit_bm25(self, corpus_text):
    print(f"📊 Fitting BM25 on {len(corpus_text)} chunks...")
    self.bm25_ef.fit(corpus_text)

  def create_hybrid_collection(self, name):
    if self.client.has_collection(name):
      self.client.drop_collection(name)
      
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="doc_name", datatype=DataType.VARCHAR, max_length=200)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = self.client.prepare_index_params()
    index_params.add_index(field_name="dense_vector", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
    index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={"drop_ratio_build": 0.2})

    self.client.create_collection(collection_name=name, schema=schema, index_params=index_params)

  def process_and_index(self, collection_name, docs):
    print(f"\n🚀 Processing {collection_name}...")
    
    for d in docs: self.doc_store[d['doc_name']] = d['text']
    
    if self.client.has_collection(collection_name):
      try:
        self.client.load_collection(collection_name)
        res = self.client.query(collection_name, filter="", limit=1)
        if res:
          print(f"✅ Collection '{collection_name}' already exists. Re-fitting BM25...")
          temp_chunks = []
          for d in docs[:500]: 
              clean_t = TextProcessor.super_clean(d['text'])
              temp_chunks.extend(clean_t.split()[:512]) 
          self.fit_bm25(temp_chunks) # Fit con caché
          return 
      except:
        self.client.drop_collection(collection_name)

    all_chunks = []
    doc_names = []
    
    for doc in tqdm(docs, desc="Limpiando y Fragmentando"):
      clean_text = TextProcessor.super_clean(doc['text'])
      chunks = TextProcessor.chunk_text(clean_text)
      for chunk in chunks:
        all_chunks.append(chunk)
        doc_names.append(doc['doc_name'])

    self.fit_bm25(all_chunks)
    self.create_hybrid_collection(collection_name)
    
    batch_size = 50
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexing"):
      batch_text = all_chunks[i:i+batch_size]
      batch_names = doc_names[i:i+batch_size]
      
      dense_vecs = self.dense_model.encode(batch_text, normalize_embeddings=True)
      sparse_matrix = self.bm25_ef.encode_documents(batch_text)
      
      rows = []
      for j in range(len(batch_text)):
        start = sparse_matrix.indptr[j]
        end = sparse_matrix.indptr[j+1]
        indices = sparse_matrix.indices[start:end]
        data = sparse_matrix.data[start:end]
        
        sparse_dict = {int(idx): float(val) for idx, val in zip(indices, data)}
        if not sparse_dict: sparse_dict = {0: 0.00001}
        
        rows.append({
          "doc_name": batch_names[j],
          "dense_vector": dense_vecs[j],
          "sparse_vector": sparse_dict,
          "text_preview": batch_text[j][:100]
        })
      self.client.insert(collection_name, rows)
      
    print("⏳ Creating Index...")
    self.client.load_collection(collection_name)

  def search_asymmetric(self, collection_name, original_query, expanded_query):
    """
    KEY CHANGE: Asymmetric Querying
    - dense vector gets the expanded query (LLM context)
    - sparse vector gets the original query (prevents BM25 keyword noise)
    """
    clean_original = TextProcessor.super_clean(original_query)
    clean_expanded = TextProcessor.super_clean(expanded_query)

    # 1. Dense (InLegalBERT) understands the extra LLM keywords
    query_dense = self.dense_model.encode([clean_expanded], normalize_embeddings=True)[0]
    
    # 2. Sparse (BM25) only looks for exact matches of what the user actually typed
    sparse_matrix = self.bm25_ef.encode_queries([clean_original])
    
    start = sparse_matrix.indptr[0]
    end = sparse_matrix.indptr[1]
    indices = sparse_matrix.indices[start:end]
    data = sparse_matrix.data[start:end]
    query_sparse = {int(idx): float(val) for idx, val in zip(indices, data)}
    
    req_dense = AnnSearchRequest(data=[query_dense], anns_field="dense_vector", param={"metric_type": "COSINE", "params": {"nprobe": 10}}, limit=TOP_K)
    req_sparse = AnnSearchRequest(data=[query_sparse], anns_field="sparse_vector", param={"metric_type": "IP", "params": {"drop_ratio_search": 0.0}}, limit=TOP_K)
    
    res = self.client.hybrid_search(
      collection_name, reqs=[req_dense, req_sparse], ranker=RRFRanker(k=20), limit=TOP_K, output_fields=["doc_name"]
    )
    
    seen = set()
    results = []
    for hit in res[0]:
      name = hit['entity']['doc_name']
      if name not in seen:
        # Recuperar score original del RRF para ordenar globalmente después
        results.append((name, hit['distance']))
        seen.add(name)
    return results

def load_files(path, prefix):
  d = []
  for f in glob.glob(os.path.join(path, f"{prefix}*.txt")):
    d.append({"doc_name": os.path.splitext(os.path.basename(f))[0], "text": open(f, errors='ignore').read()})
  return d

def load_qrels(filepaths):
    qrels = {}
    for filepath in filepaths:
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4 and int(parts[3]) > 0:
                        qrels.setdefault(parts[0], set()).add(parts[2])
        except FileNotFoundError: pass
    return qrels

def calculate_metrics(retrieved_docs, relevant_docs, k=K_METRICS):
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = [doc for doc in retrieved_k if doc in relevant_docs]
    
    precision = len(relevant_retrieved) / k if k > 0 else 0.0
    recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0.0
    
    ap, hits = 0.0, 0
    for i, doc in enumerate(retrieved_k):
        if doc in relevant_docs:
            hits += 1
            ap += hits / (i + 1)
            
    ap = ap / len(relevant_docs) if relevant_docs else 0.0
    return precision, recall, ap

def main():
  cases = load_files(PATH_CASES, "C")
  statutes = load_files(PATH_STATUTES, "S")
  queries = []
  with open(PATH_QUERIES, errors='ignore') as f:
    for l in f: 
      if "||" in l: queries.append({"id": l.split("||")[0], "text": l.split("||")[1].strip()})

  print("📂 Cargando juicios de relevancia...")
  qrels = load_qrels([PATH_QRELS_STATUTES, PATH_QRELS_CASES])

  agent = AgenticLayer()  
  engine = CustomHybridSearch()
  
  # NEW V8 COLLECTIONS
  engine.process_and_index("aila_cases_v8", cases)
  engine.process_and_index("aila_statutes_v8", statutes)
  
  with open(TREC_OUTPUT_FILE, "w") as f: f.write("")
  with open(METRICS_OUTPUT_FILE, "w") as f: f.write("")

  print(f"\n🚀 Running ASYMMETRIC Agentic RAG on {len(queries)} queries...")
  
  total_precision, total_recall, total_ap, queries_evaluated = 0.0, 0.0, 0.0, 0
  
  for q in tqdm(queries):
    q_id = q['id']
    original_query = q['text']
    
    # Expand query with LLM
    legal_keywords = agent.get_legal_keywords(original_query)
    expanded_query = f"{original_query} {legal_keywords}"

    # Asymmetric Search
    cands_c = engine.search_asymmetric("aila_cases_v8", original_query, expanded_query)
    cands_s = engine.search_asymmetric("aila_statutes_v8", original_query, expanded_query)
    
    # Merge and Sort Globally by RRF Score
    all_cands = cands_c + cands_s
    all_cands.sort(key=lambda x: x[1], reverse=True)
    
    # Deduplicate
    seen = set()
    final = []
    for doc_id, score in all_cands:
      if doc_id not in seen:
        final.append((doc_id, score))
        seen.add(doc_id)
        if len(final) == K_METRICS: break # Solo guardamos el Top 60
    
    with open(TREC_OUTPUT_FILE, "a") as f:
        for rank, (doc_id, score) in enumerate(final):
            f.write(f"{q_id} Q0 {doc_id} {rank+1} {score:.4f} AsymmetricAgentic\n")
    
    retrieved_doc_ids = [doc[0] for doc in final]
    
    if q_id in qrels:
        p, r, ap = calculate_metrics(retrieved_doc_ids, qrels[q_id], k=K_METRICS)
        total_precision += p; total_recall += r; total_ap += ap; queries_evaluated += 1
        with open(METRICS_OUTPUT_FILE, "a") as f:
            f.write(f"QUERY: {q_id:<10} | P@{K_METRICS}: {p:.4f} | R@{K_METRICS}: {r:.4f} | AP@{K_METRICS}: {ap:.4f}\n")

  if queries_evaluated > 0:
      mean_precision = total_precision / queries_evaluated
      mean_recall = total_recall / queries_evaluated
      map_score = total_ap / queries_evaluated
      
      report = (
          f"\n{'='*50}\n"
          f"📊 REPORTE DE EVALUACIÓN V8 (Asymmetric Querying)\n"
          f"{'='*50}\n"
          f"Consultas evaluadas : {queries_evaluated}\n"
          f"Evaluado en Top-K   : {K_METRICS}\n"
          f"{'-' * 50}\n"
          f"Precision@{K_METRICS}      : {mean_precision:.4f}\n"
          f"Recall@{K_METRICS}         : {mean_recall:.4f}\n"
          f"MAP (Mean Avg Prec) : {map_score:.4f}\n"
          f"{'='*50}\n"
      )
      print(report)
      with open(METRICS_OUTPUT_FILE, "a") as f: f.write(report) 

if __name__ == "__main__":
  main()