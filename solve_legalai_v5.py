

"""
AILA-Challenge Agentic Hybrid Retrieval System (v5)
---------------------------------------------------
This program implements an advanced retrieval-augmented generation (RAG) pipeline for legal document search.
It combines dense and sparse retrieval (using BGE-M3 embeddings and BM25-like sparse vectors in Milvus)
with an agentic reasoning layer powered by Ollama (Llama3) to expand queries with legal concepts.
A cross-encoder reranker is used for final ranking.

Key differences from previous version (v4):
- Uses BGE-M3 for hybrid (dense+sparse) retrieval instead of SentenceTransformer/BM25.
- Integrates an agentic layer (Ollama/Llama3) to expand queries with legal reasoning and keywords.
- Employs Milvus hybrid search and RRF fusion for candidate retrieval.
- More modular and agentic, with improved legal context awareness in retrieval.
"""
import os
import re
import time
from tqdm import tqdm
from openai import OpenAI  # Client for Ollama
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, DataType, Function, FunctionType
# Requires: pip install "pymilvus[model]"
from pymilvus.model.hybrid import BGEM3EmbeddingFunction 
from sentence_transformers import CrossEncoder

# --- CONFIGURATION ---
BASE_DIR = "/Users/javiersantana/INF_2024_2025/Trabajo Fin de Grado/AILA-Challenge/archive"
PATH_CASES = os.path.join(BASE_DIR, "Object_casedocs")
PATH_STATUTES = os.path.join(BASE_DIR, "Object_statutes")
PATH_QUERIES = os.path.join(BASE_DIR, "Query_doc.txt")
OUTPUT_FILE = "test_results/retrieval_results_v5.txt"

# Connection Configs
MILVUS_URI = "http://localhost:19530"
OLLAMA_BASE_URL = "http://localhost:11434/v1" 
OLLAMA_MODEL = "llama3"

# Search Params
TOP_K_RETRIEVAL = 100
FINAL_K_CASES = 30
FINAL_K_STATUTES = 30

# METRICS
K_METRICS = 60
PATH_QRELS_STATUTES = os.path.join(BASE_DIR, "relevance_judgments_statutes.txt")
PATH_QRELS_CASES = os.path.join(BASE_DIR, "relevance_judgments_priorcases.txt")
TREC_OUTPUT_FILE = "test_results/v5/trec_rankings.txt"    # El formato crudo para Kaggle
METRICS_OUTPUT_FILE = "test_results/v5/eval_metrics.txt"  # Tu reporte legible de métricas

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

class RetrievalSystem:
    def __init__(self):
        print("⏳ Loading BGE-M3 (This handles Vectors AND BM25-Sparse)...")
        # BGE-M3 creates Dense vectors (Concept) and Sparse vectors (Keywords/BM25)
        self.ef = BGEM3EmbeddingFunction(use_fp16=False, device='mps')
        
        print("⏳ Loading Reranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
        
        self.client = MilvusClient(uri=MILVUS_URI)
        self.doc_store = {} # Cache for full text

    def create_hybrid_collection(self, name):
        if self.client.has_collection(name):
            self.client.drop_collection(name)
            
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="doc_name", datatype=DataType.VARCHAR, max_length=200)
        # Dense Vector (Semantic)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        # Sparse Vector (BM25 equivalent)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = self.client.prepare_index_params()
        
        # Dense Index
        index_params.add_index(field_name="dense_vector", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
        # Sparse Index (Inverted Index like Lucene/BM25)
        index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={"drop_ratio_build": 0.2})

        self.client.create_collection(collection_name=name, schema=schema, index_params=index_params)

    def process_and_index(self, collection_name, docs):
        """Checks cache, generates embeddings (Dense+Sparse), and indexes."""
        # 1. Always load Doc Store
        print(f"\n📂 Loading text for {collection_name}...")
        for d in docs: self.doc_store[d['doc_name']] = d['text']
        
        # 2. Check Cache
        if self.client.has_collection(collection_name):
            try:
                self.client.load_collection(collection_name)
                res = self.client.query(collection_name, filter="", limit=1)
                if res:
                    print(f"✅ Collection '{collection_name}' already exists and is loaded. Skipping Embeddings.")
                    return
            except:
                print(f"⚠️ Collection '{collection_name}' seems broken/empty. Re-indexing...")
                self.client.drop_collection(collection_name)

        # 3. Embed & Insert
        print(f"🧠 Generating BGE-M3 Embeddings (Dense + Sparse) for {collection_name}...")
        self.create_hybrid_collection(collection_name)
        
        all_chunks_text = []
        all_chunks_meta = []
        
        for doc in tqdm(docs, desc="Chunking"):
            words = doc['text'].split()
            for i in range(0, len(words), 512-100):
                chunk = " ".join(words[i:i+512])
                all_chunks_text.append(chunk)
                all_chunks_meta.append(doc['doc_name'])

        batch_size = 50
        for i in tqdm(range(0, len(all_chunks_text), batch_size), desc="Indexing"):
            batch_text = all_chunks_text[i:i+batch_size]
            batch_doc_names = all_chunks_meta[i:i+batch_size]
            
            # Generate BOTH vectors
            output = self.ef(batch_text)
            
            # --- FIX STARTS HERE ---
            # Access the Sparse Matrix (CSR Format) directly
            sparse_matrix = output["sparse"]
            
            rows = []
            for j in range(len(batch_text)):
                # Extract specific row data using pointers
                start = sparse_matrix.indptr[j]
                end = sparse_matrix.indptr[j+1]
                
                # Create dictionary {index: value} for Milvus
                indices = sparse_matrix.indices[start:end]
                data = sparse_matrix.data[start:end]
                sparse_dict = {int(k): float(v) for k, v in zip(indices, data)}
                
                rows.append({
                    "doc_name": batch_doc_names[j],
                    "dense_vector": output["dense"][j],
                    "sparse_vector": sparse_dict, # Pass the dictionary
                    "text_preview": batch_text[j][:200]
                })
            
            self.client.insert(collection_name, rows)

    def search_hybrid(self, collection_name, query_text):
        """Performs Milvus Hybrid Search (Dense + Sparse/BM25)"""
        # 1. Embed Query
        # This returns a dictionary with 'dense' (list) and 'sparse' (CSR Matrix)
        query_vecs = self.ef([query_text])
        
        # --- FIX: Extract Sparse Vector from Matrix manually ---
        sparse_matrix = query_vecs["sparse"]
        # Get the first (and only) row
        start = sparse_matrix.indptr[0]
        end = sparse_matrix.indptr[1]
        indices = sparse_matrix.indices[start:end]
        data = sparse_matrix.data[start:end]
        # Convert to Milvus-friendly dictionary format {index: score}
        sparse_dict = {int(k): float(v) for k, v in zip(indices, data)}
        # -------------------------------------------------------

        # 2. Search Requests
        req_dense = AnnSearchRequest(
            data=[query_vecs["dense"][0]], 
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}}, 
            limit=TOP_K_RETRIEVAL
        )
        
        req_sparse = AnnSearchRequest(
            data=[sparse_dict], # <--- Passing the fixed dictionary here
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}, 
            limit=TOP_K_RETRIEVAL
        )
        
        # 3. RRF Fusion (Combines the two results)
        res = self.client.hybrid_search(
            collection_name, 
            reqs=[req_dense, req_sparse],
            ranker=RRFRanker(k=60), 
            limit=TOP_K_RETRIEVAL, 
            output_fields=["doc_name"]
        )
        
        # Deduplicate
        seen = set()
        candidates = []
        for hit in res[0]:
            name = hit['entity']['doc_name']
            if name not in seen:
                candidates.append(name)
                seen.add(name)
        return candidates

    def rerank_sliding_window(self, query, doc_ids, top_n):
        """The 'Judge' - Reads full text with sliding window"""
        doc_scores = []
        
        for did in doc_ids:
            if did not in self.doc_store: continue
            full_text = self.doc_store[did]
            
            # Create Windows
            words = full_text.split()
            windows = [" ".join(words[i:i+400]) for i in range(0, len(words), 350)]
            
            # Optimize: Check first 3 and last 3 windows (Facts + Verdict)
            check_wins = windows[:3] + windows[-3:] if len(windows) > 6 else windows
            if not check_wins: continue
            
            pairs = [[query, w] for w in check_wins]
            scores = self.reranker.predict(pairs)
            best_score = max(scores)
            
            doc_scores.append((did, best_score))
            
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_n]

# --- HELPERS ---
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

def calculate_metrics(retrieved_docs, relevant_docs, k=K_METRICS):
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
    # 1. Load Data
    cases = load_files(PATH_CASES, "C")
    statutes = load_files(PATH_STATUTES, "S")
    queries = []
    with open(PATH_QUERIES, errors='ignore') as f:
        for l in f: 
            if "||" in l: queries.append({"id": l.split("||")[0], "text": l.split("||")[1].strip()})

    qrels = load_qrels([PATH_QRELS_STATUTES, PATH_QRELS_CASES])

    # 2. Initialize Systems
    agent = AgenticLayer()
    db = RetrievalSystem()
    
    # 3. Index (Auto-skips if cached)
    db.process_and_index("aila_cases_v5", cases)
    db.process_and_index("aila_statutes_v5", statutes)
    # 4. Limpiar archivos de salida
    with open(TREC_OUTPUT_FILE, "w") as f: f.write("")
    with open(METRICS_OUTPUT_FILE, "w") as f: f.write("")

    print(f"\n🚀 Running Agentic RAG on {len(queries)} queries...")
    
    # 5. Variables para promedios
    total_precision = 0.0
    total_recall = 0.0
    total_ap = 0.0
    queries_evaluated = 0

    for q in tqdm(queries):
        q_id = q['id']
        original_query = q['text']

        # A. Reasoning Layer: Expand query with legal keywords
        enhanced_query = agent.expand_query_with_reasoning(original_query)

        # B. Retrieval
        cands_c = db.search_hybrid("aila_cases_v5", enhanced_query)
        cands_s = db.search_hybrid("aila_statutes_v5", enhanced_query)

        # C. Reranking
        ranked_c = db.rerank_sliding_window(enhanced_query, cands_c, FINAL_K_CASES)
        ranked_s = db.rerank_sliding_window(enhanced_query, cands_s, FINAL_K_STATUTES)

        # D. Merge & Save Rankings
        final = ranked_c + ranked_s
        final.sort(key=lambda x: x[1], reverse=True)

        # --- ARCHIVO 1: Formato Oficial TREC ---
        with open(TREC_OUTPUT_FILE, "a") as f:
            for rank, (doc_id, score) in enumerate(final):
                f.write(f"{q_id} Q0 {doc_id} {rank+1} {score:.4f} AgenticHybrid\n")

        # --- ARCHIVO 2: Cálculo y guardado de Métricas ---
        retrieved_doc_ids = [doc[0] for doc in final]

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

    # 6. Guardar e imprimir reporte estandarizado al final
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

if __name__ == "__main__":
    import glob
    main()