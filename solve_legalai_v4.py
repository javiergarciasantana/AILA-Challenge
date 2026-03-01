"""
AILA-Challenge Hybrid Retrieval System (v4.1 - Milvus RRF + Optional Rerankers)
---------------------------------------------------
This version combines dense (InLegalBERT) and sparse (BM25EmbeddingFunction) 
retrieval natively inside Milvus using Reciprocal Rank Fusion (RRF).
- INCLUDES both Standard and Sliding Window Rerankers (Togglable in main).
- Uses robust text cleaning (super_clean).
- Safe chunking (300 words) to avoid token truncation.
"""

import os
import glob
import re
import unicodedata
from tqdm import tqdm
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker
from sentence_transformers import SentenceTransformer, CrossEncoder

from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer

# --- CONFIGURATION ---
BASE_DIR = "/Users/javiersantana/INF_2024_2025/Trabajo Fin de Grado/AILA-Challenge/archive"
PATH_CASES = os.path.join(BASE_DIR, "Object_casedocs")
PATH_STATUTES = os.path.join(BASE_DIR, "Object_statutes")
PATH_QUERIES = os.path.join(BASE_DIR, "Query_doc.txt")

MILVUS_URI = "http://localhost:19530"
EMBEDDING_MODEL = "./models/InLegalBERT-AILA-Tuned"  
RERANKER_MODEL = "cross-encoder/nli-deberta-v3-base"

# METRICS
os.makedirs("test_results/v4", exist_ok=True)
PATH_QRELS_STATUTES = os.path.join(BASE_DIR, "relevance_judgments_statutes.txt")
PATH_QRELS_CASES = os.path.join(BASE_DIR, "relevance_judgments_priorcases.txt")
TREC_OUTPUT_FILE = "test_results/v4_rerank_window_deberta/trec_rankings.txt"
METRICS_OUTPUT_FILE = "test_results/v4_rerank_window_deberta/eval_metrics.txt"

# HYPERPARAMETERS
CHUNK_SIZE = 300
OVERLAP = 50
TOP_K_RETRIEVAL = 100
FINAL_K_CASES = 30
FINAL_K_STATUTES = 30
K_METRICS = 60

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

class SearchSystem:
    def __init__(self):
        print("⏳ Cargando InLegalBERT...")
        self.dense_model = SentenceTransformer(EMBEDDING_MODEL)
        self.dense_dim = 768
        
        print("⏳ Inicializando BM25 Embedding Function...")
        self.analyzer = build_default_analyzer(language="en")
        self.bm25_ef = BM25EmbeddingFunction(self.analyzer)
        
        print("⏳ Cargando Cross-Encoder Reranker...")
        # Forzamos max_length=512 para simular el truncamiento estándar del modelo
        self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        
        self.client = MilvusClient(uri=MILVUS_URI)
        self.doc_store = {} # Memoria para guardar el texto original de los documentos

    def fit_bm25(self, corpus_text):
        print(f"📊 Ajustando modelo BM25 con {len(corpus_text)} fragmentos...")
        self.bm25_ef.fit(corpus_text)

    def index_collection(self, documents, collection_name):
        print(f"\nPreparando '{collection_name}'...")

        # 1. SIEMPRE procesar textos para memoria local (BM25 y Reranker lo necesitan)
        all_chunks = []
        for doc in tqdm(documents, desc=f"Cargando textos para {collection_name}"):
            # Llenar la memoria para el Reranker
            self.doc_store[doc['doc_name']] = doc['text'] 
            
            clean_text = TextProcessor.super_clean(doc['text'])
            chunks = TextProcessor.chunk_text(clean_text)
            for chunk in chunks:
                all_chunks.append({"doc_name": doc['doc_name'], "text": chunk})

        text_corpus = [x['text'] for x in all_chunks]
        
        # Ajustar BM25 siempre, aunque la colección ya exista en Milvus
        self.fit_bm25(text_corpus)

        # 2. Comprobar Milvus (Caché de vectores)
        if self.client.has_collection(collection_name):
            print(f"✅ Colección '{collection_name}' encontrada en Milvus. Omitiendo vectorización.")
            self.client.load_collection(collection_name)
            return # Salimos aquí, nos ahorramos el trabajo pesado

        # 3. Si NO existe en Milvus, creamos e insertamos (Solo pasa la primera vez)
        print(f"⚠️ Colección no encontrada. Creando e indexando vectores...")
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="doc_name", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
        index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={"drop_ratio_build": 0.2})

        self.client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

        batch_size = 32
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Vectorizando e Insertando"):
            batch = all_chunks[i:i+batch_size]
            batch_texts = [x['text'] for x in batch]
            
            dense_vecs = self.dense_model.encode(batch_texts, normalize_embeddings=True)
            sparse_matrix = self.bm25_ef.encode_documents(batch_texts)
            
            payload = []
            for j in range(len(batch_texts)):
                start, end = sparse_matrix.indptr[j], sparse_matrix.indptr[j+1]
                indices, data = sparse_matrix.indices[start:end], sparse_matrix.data[start:end]
                sparse_dict = {int(idx): float(val) for idx, val in zip(indices, data)}
                if not sparse_dict: sparse_dict = {0: 0.00001}
                
                payload.append({
                    "doc_name": batch[j]['doc_name'],
                    "dense_vector": dense_vecs[j],
                    "sparse_vector": sparse_dict
                })
            self.client.insert(collection_name=collection_name, data=payload)
            
        print(f"⏳ Cargando '{collection_name}' recién creada en memoria de Milvus...")
        self.client.load_collection(collection_name)

    def search_collection(self, query_text, collection_name, limit_k):
        q_dense = self.dense_model.encode([query_text], normalize_embeddings=True)[0]
        
        sparse_matrix = self.bm25_ef.encode_queries([query_text])
        start, end = sparse_matrix.indptr[0], sparse_matrix.indptr[1]
        indices, data = sparse_matrix.indices[start:end], sparse_matrix.data[start:end]
        q_sparse = {int(idx): float(val) for idx, val in zip(indices, data)}
        if not q_sparse: q_sparse = {0: 0.00001}

        req_dense = AnnSearchRequest(data=[q_dense], anns_field="dense_vector", param={"metric_type": "COSINE", "params": {"nprobe": 10}}, limit=TOP_K_RETRIEVAL)
        req_sparse = AnnSearchRequest(data=[q_sparse], anns_field="sparse_vector", param={"metric_type": "IP", "params": {"drop_ratio_search": 0.0}}, limit=TOP_K_RETRIEVAL)

        res = self.client.hybrid_search(
            collection_name, reqs=[req_dense, req_sparse], ranker=RRFRanker(k=60), limit=TOP_K_RETRIEVAL, output_fields=["doc_name"]
        )

        seen = set()
        results = []
        for hit in res[0]:
            name = hit['entity']['doc_name']
            if name not in seen:
                results.append((name, hit['distance'])) 
                seen.add(name)
        
        return results[:limit_k]

    def search_balanced(self, query_text):
        q_clean = TextProcessor.super_clean(query_text)
        final_cases = self.search_collection(q_clean, "aila_cases_v4", FINAL_K_CASES)
        final_statutes = self.search_collection(q_clean, "aila_statutes_v4", FINAL_K_STATUTES)

        merged_results = []
        max_len = max(len(final_cases), len(final_statutes))
        for i in range(max_len):
            if i < len(final_cases): merged_results.append(final_cases[i])
            if i < len(final_statutes): merged_results.append(final_statutes[i])
                
        return merged_results

    # =========================================================
    # NUEVAS FUNCIONES DE RE-RANKING PARA EVALUACIÓN EN EL TFG
    # =========================================================
    
    def rerank_standard(self, query, doc_ids, top_n):
        """
        Reranker Estándar: Intenta evaluar el documento entero.
        El modelo Cross-Encoder truncará automáticamente a 512 tokens (su max_length).
        Esto simula el problema del 'Lost in the middle'.
        """
        doc_scores = []
        for did in doc_ids:
            if did not in self.doc_store: continue
            full_text = self.doc_store[did]
            
            # El modelo trunca por debajo sin avisar
            score = self.reranker.predict([query, full_text])
            doc_scores.append((did, score))
            
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_n]

    def rerank_sliding_window(self, query, doc_ids, top_n):
        """
        Reranker Optimizado (Sliding Windows): 
        Divide el texto para no perder contexto y evalúa hechos (inicio) y veredicto (final).
        """
        doc_scores = []
        for did in doc_ids:
            if did not in self.doc_store: continue
            full_text = self.doc_store[did]
            
            words = full_text.split()
            # Ventanas de 400 palabras con salto de 350 (solapamiento de 50)
            windows = [" ".join(words[i:i+400]) for i in range(0, len(words), 350)]
            
            # Filtrar inicio y fin (donde suele estar lo importante en derecho)
            check_wins = windows[:3] + windows[-3:] if len(windows) > 6 else windows
            if not check_wins: continue
            
            # Predecir cada ventana y quedarse con la puntuación máxima
            pairs = [[query, w] for w in check_wins]
            scores = self.reranker.predict(pairs)
            
            # FIX: Si el modelo devuelve 3 columnas (NLI), extraemos la de "Entailment"
            if len(scores.shape) > 1:
                # Asumimos que la Implicación (Entailment) es la columna 1 o la que tenga el mayor valor general
                scores = scores[:, 1] # Extrae solo la segunda columna de todas las ventanas            
           
            best_score = float(max(scores))
            doc_scores.append((did, best_score))
            
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_n]


# --- FUNCIONES DE EVALUACIÓN Y CARGA ---
def load_data(path, prefix):
    docs = []
    for f in glob.glob(os.path.join(path, f"{prefix}*.txt")):
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
    qrels = {}
    for filepath in filepaths:
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4 and int(parts[3]) > 0:
                        qrels.setdefault(parts[0], set()).add(parts[2])
        except: pass
    return qrels

def calculate_metrics(retrieved_docs, relevant_docs, k):
    retrieved_k = retrieved_docs[:k]
    hits = [doc for doc in retrieved_k if doc in relevant_docs]
    p = len(hits) / k if k > 0 else 0.0
    r = len(hits) / len(relevant_docs) if relevant_docs else 0.0
    ap, hit_count = 0.0, 0
    for i, doc in enumerate(retrieved_k):
        if doc in relevant_docs:
            hit_count += 1
            ap += hit_count / (i + 1)
    ap = ap / len(relevant_docs) if relevant_docs else 0.0
    return p, r, ap

def main():
    cases = load_data(PATH_CASES, "C")
    statutes = load_data(PATH_STATUTES, "S")
    queries = load_queries(PATH_QUERIES)
    qrels = load_qrels([PATH_QRELS_STATUTES, PATH_QRELS_CASES])

    if not cases: return

    system = SearchSystem()
    system.index_collection(cases, "aila_cases_v4")
    system.index_collection(statutes, "aila_statutes_v4")

    print(f"\n🚀 Lanzando Búsqueda Híbrida Pura en {len(queries)} consultas...")
    
    total_p, total_r, total_ap, eval_count = 0, 0, 0, 0
    
    with open(TREC_OUTPUT_FILE, "w") as f_trec, open(METRICS_OUTPUT_FILE, "w") as f_metrics:
        for q in tqdm(queries):
            q_id, q_text = q['id'], q['text']
            
            # 1. Recuperación Inicial (Milvus RRF Híbrido)
            final_results = system.search_balanced(q_text)

            # -------------------------------------------------------------
            # ZONA DE EXPERIMENTOS: 
            # Descomenta una de estas dos líneas para activar el Re-ranker.
            # Si ambas están comentadas, usarás el RRF puro de Milvus.
            # -------------------------------------------------------------
            
            candidate_ids = [doc_id for doc_id, _ in final_results]
            
            #final_results = system.rerank_standard(q_text, candidate_ids, K_METRICS)
            final_results_2 = system.rerank_sliding_window(q_text, candidate_ids, K_METRICS)
            
            # -------------------------------------------------------------

            # Archivo TREC
            for rank, (doc_id, score) in enumerate(final_results_2):
                f_trec.write(f"{q_id} Q0 {doc_id} {rank+1} {score:.4f} MilvusHybrid\n")

            # Métricas
            retrieved_ids = [doc_id for doc_id, _ in final_results_2]
            if q_id in qrels:
                p, r, ap = calculate_metrics(retrieved_ids, qrels[q_id], K_METRICS)
                total_p += p; total_r += r; total_ap += ap; eval_count += 1
                f_metrics.write(f"QUERY: {q_id:<10} | P@{K_METRICS}: {p:.4f} | R@{K_METRICS}: {r:.4f} | MAP@{K_METRICS}: {ap:.4f}\n")

        # Reporte Final
        if eval_count > 0:
            report = (
                f"\n{'='*50}\n"
                f"📊 REPORTE GLOBAL\n"
                f"{'='*50}\n"
                f"Consultas evaluadas : {eval_count}\n"
                f"Precision@{K_METRICS}      : {total_p/eval_count:.4f}\n"
                f"Recall@{K_METRICS}         : {total_r/eval_count:.4f}\n"
                f"MAP@{K_METRICS}            : {total_ap/eval_count:.4f}\n"
                f"{'='*50}\n"
            )
            print(report)
            f_metrics.write(report)

if __name__ == "__main__":
    main()