"""
AILA-Challenge Agentic Hybrid Retrieval System (v4)
---------------------------------------------------
This program implements a retrieval-augmented pipeline for legal document search.
It combines dense retrieval (using SentenceTransformer embeddings stored in Milvus)
and sparse retrieval (BM25) for candidate generation, followed by cross-encoder reranking.

Key features:
- Uses InLegalBERT for dense embeddings and BM25 for sparse retrieval.
- Stores dense vectors in Milvus for efficient ANN search.
- Cross-encoder reranker (MiniLM-L-12) for final ranking.
- Balanced retrieval of both case law and statutes.
- Modular design for easy extension.

This is version 4. Key differences from previous versions:
- Improved chunking and reranking strategies.
- More robust candidate merging and output formatting.
"""

import os
import sys
import glob
import re
import numpy as np
import time
from tqdm import tqdm
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# --- CONFIGURATION ---
BASE_DIR = "/Users/javiersantana/INF_2024_2025/Trabajo Fin de Grado/AILA-Challenge/archive"
PATH_CASES = os.path.join(BASE_DIR, "Object_casedocs")
PATH_STATUTES = os.path.join(BASE_DIR, "Object_statutes")
PATH_QUERIES = os.path.join(BASE_DIR, "Query_doc.txt")
OUTPUT_FILE = "test_results/retrieval_results_v4.txt"

MILVUS_URI = "http://localhost:19530"

# MODELS
EMBEDDING_MODEL = "./models/InLegalBERT-AILA-Tuned"  
# Using the Larger (L-12) Cross Encoder for better accuracy
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# METRICS
PATH_QRELS_STATUTES = os.path.join(BASE_DIR, "relevance_judgments_statutes.txt")
PATH_QRELS_CASES = os.path.join(BASE_DIR, "relevance_judgments_priorcases.txt")
K_METRICS = 60
TREC_OUTPUT_FILE = "test_results/v4/trec_rankings.txt"    # El formato crudo para Kaggle
METRICS_OUTPUT_FILE = "test_results/v4/eval_metrics.txt"  # Tu reporte legible de métricas

# HYPERPARAMETERS
CHUNK_SIZE = 512
OVERLAP = 128
TOP_K_CANDIDATES = 100  # Candidates per collection
TOP_N_RERANK = 30       # Rerank top 30 from EACH list
FINAL_K_CASES = 30       # Force 30 Cases
FINAL_K_STATUTES = 30    # Force 30 Statutes

class TextProcessor:
    @staticmethod
    def clean(text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text

    @staticmethod
    def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
        words = text.split()
        if not words: return []
        chunks = []
        for i in range(0, len(words), size - overlap):
            chunk = " ".join(words[i:i + size])
            chunks.append(chunk)
        return chunks

class SearchSystem:
    def __init__(self):
        print("⏳ Loading Powerful Models...")
        self.bi_encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.cross_encoder = CrossEncoder(RERANKER_MODEL, max_length=512)
        self.client = MilvusClient(uri=MILVUS_URI)
        
        self.bm25_cases = None
        self.bm25_statutes = None
        self.doc_store = {} 
        self.map_cases = {}
        self.map_statutes = {}

    def index_collection(self, documents, collection_name, is_statute=False):
        """Standard Indexing with Cache Check"""
        print(f"\nPreparing '{collection_name}' ({len(documents)} docs)...")
        
        all_chunks = []
        bm25_corpus = []
        
        # Load text & prepare chunks (Always needed for Reranker/BM25)
        for doc in tqdm(documents, desc="Loading Text"):
            self.doc_store[doc['doc_name']] = doc['text']
            chunks = TextProcessor.chunk_text(doc['text'])
            for chunk in chunks:
                all_chunks.append({"doc_name": doc['doc_name'], "text": chunk})
                bm25_corpus.append(TextProcessor.clean(chunk).split())

        # Train BM25
        print(f"Training BM25...")
        bm25_index = BM25Okapi(bm25_corpus)
        current_map = {i: c['doc_name'] for i, c in enumerate(all_chunks)}
        
        if is_statute:
            self.bm25_statutes = bm25_index
            self.map_statutes = current_map
        else:
            self.bm25_cases = bm25_index
            self.map_cases = current_map

        # Check Milvus Cache
        if self.client.has_collection(collection_name):
            res = self.client.query(collection_name=collection_name, filter="", limit=1)
            if res:
                print(f"✅ Collection '{collection_name}' cached. Skipping Vectors.")
                return

        # Generate Vectors
        print(f"Generating Vectors for '{collection_name}'...")
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
        
        self.client.create_collection(
            collection_name=collection_name,
            dimension=768, 
            metric_type="COSINE",
            auto_id=True
        )

        batch_size = 64
        #For loop with progress bar
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Vectors"):
            batch = all_chunks[i : i + batch_size]
            texts = [x['text'] for x in batch]
            vectors = self.bi_encoder.encode(texts, normalize_embeddings=True)
            
            payload = []
            for idx, vec in enumerate(vectors):
                payload.append({
                    "vector": vec,
                    "doc_name": batch[idx]['doc_name'],
                    "text_preview": batch[idx]['text'][:100]
                })
            self.client.insert(collection_name=collection_name, data=payload)

    def get_candidates(self, query_text, collection_name, bm25_index, bm25_map):
        candidates = set()
        
        # 1. BM25
        q_tokens = TextProcessor.clean(query_text).split()
        bm25_scores = bm25_index.get_scores(q_tokens)
        top_bm25 = np.argsort(bm25_scores)[::-1][:TOP_K_CANDIDATES]
        for idx in top_bm25:
            candidates.add(bm25_map[idx])

        # 2. Milvus
        q_vector = self.bi_encoder.encode([query_text], normalize_embeddings=True)
        milvus_res = self.client.search(
            collection_name=collection_name,
            data=q_vector,
            limit=TOP_K_CANDIDATES,
            output_fields=["doc_name"]
        )
        for hit in milvus_res[0]:
            candidates.add(hit['entity']['doc_name'])
            
        return list(candidates)

    def rerank_list(self, query, candidate_ids):
        """Scores a list of doc IDs using Sliding Window Cross-Encoder"""
        scores = []
        for doc_id in candidate_ids:
            if doc_id not in self.doc_store: continue
            
            full_text = self.doc_store[doc_id]
            # Max score strategy
            windows = TextProcessor.chunk_text(full_text, size=400, overlap=50)
            
            # Optimization: First 3 + Last 2 windows
            if len(windows) > 5:
                check_windows = windows[:3] + windows[-2:]
            else:
                check_windows = windows
                
            pairs = [[query, w] for w in check_windows]
            if not pairs: continue
            
            w_scores = self.cross_encoder.predict(pairs)
            best_score = max(w_scores)
            scores.append((doc_id, best_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def search_balanced(self, query_id, query_text):
        print(f"\n🔍 Query {query_id}...")
        
        # 1. Retrieve Cases
        cand_cases = self.get_candidates(query_text, "aila_cases_v4", self.bm25_cases, self.map_cases)
        # 2. Retrieve Statutes
        cand_statutes = self.get_candidates(query_text, "aila_statutes_v4", self.bm25_statutes, self.map_statutes)

        print(f"   Found {len(cand_cases)} Cases, {len(cand_statutes)} Statutes")
        
        # 3. Rerank Cases
        ranked_cases = self.rerank_list(query_text, cand_cases[:TOP_N_RERANK])
        final_cases = ranked_cases[:FINAL_K_CASES]
        
        # 4. Rerank Statutes
        ranked_statutes = self.rerank_list(query_text, cand_statutes[:TOP_N_RERANK])
        final_statutes = ranked_statutes[:FINAL_K_STATUTES]
        
        # 5. Merge
        # We interleave them: Case, Statute, Case, Statute... to ensure diversity
        merged_results = []
        max_len = max(len(final_cases), len(final_statutes))
        
        for i in range(max_len):
            if i < len(final_cases):
                merged_results.append(final_cases[i])
            if i < len(final_statutes):
                merged_results.append(final_statutes[i])
                
        print(f"   🏆 Top Case: {final_cases[0] if final_cases else 'None'}")
        print(f"   🏆 Top Statute: {final_statutes[0] if final_statutes else 'None'}")
        
        return merged_results

def load_data(path, prefix):
    docs = []
    files = glob.glob(os.path.join(path, f"{prefix}*.txt"))
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        with open(f, 'r', errors='ignore') as file:
            docs.append({"doc_name": name, "text": file.read()})
    return docs

def load_queries(path):
    queries = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if "||" in line:
                p = line.strip().split("||")
                queries.append({"id": p[0], "text": p[1]})
    return queries

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
    cases = load_data(PATH_CASES, "C")
    statutes = load_data(PATH_STATUTES, "S")
    queries = load_queries(PATH_QUERIES)
    
    qrels = load_qrels([PATH_QRELS_STATUTES, PATH_QRELS_CASES])

    if not cases: return

    system = SearchSystem()
    system.index_collection(cases, "aila_cases_v4", is_statute=False)
    system.index_collection(statutes, "aila_statutes_v4", is_statute=True)

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


        # B. Retrieval
        final = system.search_balanced(q["id"], q["text"])

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