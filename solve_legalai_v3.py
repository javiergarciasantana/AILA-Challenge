import os
import re
import sys
import glob
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# --- CONFIGURACIÓN ---
BASE_DIR = "/Users/javiersantana/INF_2024_2025/Trabajo Fin de Grado/AILA-Challenge/archive"
PATH_CASES = os.path.join(BASE_DIR, "Object_casedocs")
PATH_STATUTES = os.path.join(BASE_DIR, "Object_statutes")
PATH_QUERIES = os.path.join(BASE_DIR, "Query_doc.txt")
OUTPUT_FILE = "test_results/retrieval_results_v3.txt"

#METRICS
TREC_FILE = "test_results/v3/trec_rankings.txt"
QRELS_FILES = [
    "archive/relevance_judgments_statutes.txt",
    "archive/relevance_judgments_priorcases.txt"
]
OUTPUT_METRICS = "test_results/v3/eval_metrics.txt"
K_METRICS = 50

MILVUS_URI = "http://localhost:19530"
DIMENSION = 768
# Modelo optimizado para leyes
EMBEDDING_MODEL = "law-ai/InLegalBERT" # Si falla, usa "BAAI/bge-base-en-v1.5"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
TOP_K_RETRIEVAL = 100
RRF_K = 60

class TextProcessor:
    @staticmethod
    def clean(text):
        # Limpieza suave para no borrar números de leyes
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text

    @staticmethod
    def extract_legal_keywords(text):
        """
        Intenta extraer menciones a leyes (Section X, Act Y) para potenciar BM25.
        """
        # Patrón para capturar "Section 302", "Article 14", "Penal Code"
        pattern = r"(section\s+\d+|article\s+\d+|act\s+\d{4}|penal\s+code|constitution)"
        matches = re.findall(pattern, text.lower())
        if matches:
            # Devuelve una cadena con las leyes repetidas para darles peso
            return " ".join(matches * 3) + " " + text
        return text

    @staticmethod
    def chunk_text(text, doc_name="DEBUG"):
        words = text.split()
        if not words: return []
        chunks = []
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = " ".join(words[i:i + CHUNK_SIZE])
            chunks.append(chunk)
        return chunks

class SearchSystem:
    def __init__(self):
        print("⏳ Cargando InLegalBERT (Modelo específico legal)...")
        # InLegalBERT es mucho mejor que BGE para leyes indias/británicas
        try:
            self.encoder = SentenceTransformer("law-ai/InLegalBERT") 
            self.dim = 768
        except:
            print("⚠️ InLegalBERT falló, usando BGE-Base...")
            self.encoder = SentenceTransformer("BAAI/bge-base-en-v1.5")
            self.dim = 768
            
        self.client = MilvusClient(uri=MILVUS_URI)
        self.bm25_cases = None
        self.bm25_statutes = None
        self.map_cases = {}
        self.map_statutes = {}

    def index_collection(self, documents, collection_name, is_statute=False):
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
        
        self.client.create_collection(
            collection_name=collection_name,
            dimension=self.dim,
            metric_type="COSINE",
            auto_id=True
        )

        all_chunks = []
        bm25_corpus = []
        
        for doc in tqdm(documents, desc=f"Indexando {collection_name}"):
            chunks = TextProcessor.chunk_text(doc['text'])
            for chunk in chunks:
                all_chunks.append({"doc_name": doc['doc_name'], "text": chunk})
                bm25_corpus.append(TextProcessor.clean(chunk).split())

        # BM25
        bm25_index = BM25Okapi(bm25_corpus)
        current_map = {i: c['doc_name'] for i, c in enumerate(all_chunks)}
        
        if is_statute:
            self.bm25_statutes = bm25_index
            self.map_statutes = current_map
        else:
            self.bm25_cases = bm25_index
            self.map_cases = current_map

        # Milvus
        batch_size = 32
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Vectores"):
            batch = all_chunks[i : i + batch_size]
            texts = [x['text'] for x in batch]
            vectors = self.encoder.encode(texts, normalize_embeddings=True)
            
            payload = []
            for idx, vec in enumerate(vectors):
                payload.append({
                    "vector": vec,
                    "doc_name": batch[idx]['doc_name'],
                    "text_preview": batch[idx]['text'][:100]
                })
            self.client.insert(collection_name=collection_name, data=payload)

    def reciprocal_rank_fusion(self, doc_ranks):
        final_scores = {}
        for doc_id, ranks in doc_ranks.items():
            rrf_score = 0
            for r in ranks:
                rrf_score += 1 / (RRF_K + r)
            final_scores[doc_id] = rrf_score
        return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    def search(self, query_text):
        # 1. ENRIQUECIMIENTO DE QUERY
        # Detectamos leyes explícitas y las añadimos al principio para BM25
        query_enriched = TextProcessor.extract_legal_keywords(query_text)
        q_tokens = TextProcessor.clean(query_enriched).split()
        
        # Para vectores usamos la query original (semántica)
        q_vector = self.encoder.encode([query_text], normalize_embeddings=True)
        
        doc_ranks = {} 
        def add_ranking(results_list):
            for rank, doc_id in enumerate(results_list):
                if doc_id not in doc_ranks: doc_ranks[doc_id] = []
                doc_ranks[doc_id].append(rank + 1)

        # A. CASOS
        bm25_scores = self.bm25_cases.get_scores(q_tokens)
        top_idx = np.argsort(bm25_scores)[::-1][:TOP_K_RETRIEVAL]
        add_ranking([self.map_cases[i] for i in top_idx])

        milvus_res = self.client.search(collection_name="aila_cases", data=q_vector, limit=TOP_K_RETRIEVAL, output_fields=["doc_name"])
        add_ranking([hit['entity']['doc_name'] for hit in milvus_res[0]])

        # B. ESTATUTOS
        bm25_scores_s = self.bm25_statutes.get_scores(q_tokens)
        top_s_idx = np.argsort(bm25_scores_s)[::-1][:TOP_K_RETRIEVAL]
        add_ranking([self.map_statutes[i] for i in top_s_idx])

        milvus_res_s = self.client.search(collection_name="aila_statutes", data=q_vector, limit=TOP_K_RETRIEVAL, output_fields=["doc_name"])
        add_ranking([hit['entity']['doc_name'] for hit in milvus_res_s[0]])

        # C. FUSIÓN
        return self.reciprocal_rank_fusion(doc_ranks)[:10]

def load_docs(path, prefix):
    docs = []
    files = glob.glob(os.path.join(path, f"{prefix}*.txt"))
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        with open(f, 'r', errors='ignore') as file:
            docs.append({"doc_name": name, "text": file.read()})
    return docs

def load_queries(path):
    queries = []
    if os.path.exists(path):
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

def load_qrels(filepaths):
    qrels = {}
    for filepath in filepaths:
        if not os.path.exists(filepath):
            continue
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                    if rel > 0:
                        qrels.setdefault(qid, set()).add(docid)
    return qrels

def load_trec_rankings(filepath):
    rankings = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, docid, rank, score, _ = parts
                rankings.setdefault(qid, []).append(docid)
    return rankings

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    relevant_retrieved = [doc for doc in retrieved_k if doc in relevant]
    return len(relevant_retrieved) / k if k else 0.0

def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    relevant_retrieved = [doc for doc in retrieved_k if doc in relevant]
    return len(relevant_retrieved) / len(relevant) if relevant else 0.0

def average_precision(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = 0
    sum_precisions = 0.0
    for i, doc in enumerate(retrieved_k):
        if doc in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(relevant) if relevant else 0.0

def main():
    qrels = load_qrels(QRELS_FILES)
    rankings = load_trec_rankings(TREC_FILE)
    total_p, total_r, total_ap, n = 0, 0, 0, 0

    with open(OUTPUT_METRICS, "w") as fout:
        for qid, retrieved in rankings.items():
            relevant = qrels.get(qid, set())
            p = precision_at_k(retrieved, relevant, K_METRICS)
            r = recall_at_k(retrieved, relevant, K_METRICS)
            ap = average_precision(retrieved, relevant, K_METRICS)
            if relevant:
                n += 1
                total_p += p
                total_r += r
                total_ap += ap
                fout.write(f"QUERY: {qid:<10} | Precision@{K_METRICS}: {p:.4f} | Recall@{K_METRICS}: {r:.4f} | AP@{K_METRICS}: {ap:.4f}\n")
            else:
                fout.write(f"QUERY: {qid:<10} | [Sin evaluación - Falta en Ground Truth]\n")
        if n > 0:
            mean_p = total_p / n
            mean_r = total_r / n
            mean_ap = total_ap / n
            report = (
                f"\n{'='*50}\n"
                f"📊 REPORTE DE EVALUACIÓN GLOBAL\n"
                f"{'='*50}\n"
                f"Consultas evaluadas : {n}\n"
                f"Evaluado en Top-K   : {K_METRICS}\n"
                f"{'-' * 50}\n"
                f"Precision@{K_METRICS}      : {mean_p:.4f}\n"
                f"Recall@{K_METRICS}         : {mean_r:.4f}\n"
                f"MAP (Mean Avg Prec) : {mean_ap:.4f}\n"
                f"{'='*50}\n"
            )
            fout.write(report)
            print(report)

if __name__ == "__main__":
    main()