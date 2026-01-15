import os
import sys
import glob
import re
import numpy as np
from tqdm import tqdm
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# --- CONFIGURACIÓN ---
BASE_DIR = "/Users/javiersantana/INF_2024_2025/Trabajo Fin de Grado/AILA-Challenge/archive"
PATH_CASES = os.path.join(BASE_DIR, "Object_casedocs")
PATH_STATUTES = os.path.join(BASE_DIR, "Object_statutes")
PATH_QUERIES = os.path.join(BASE_DIR, "Query_doc.txt")
OUTPUT_FILE = "retrieval_results_v3.txt"

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

def main():
    cases = load_docs(PATH_CASES, "C")
    statutes = load_docs(PATH_STATUTES, "S")
    queries = load_queries(PATH_QUERIES)

    engine = SearchSystem()
    engine.index_collection(cases, "aila_cases", is_statute=False)
    engine.index_collection(statutes, "aila_statutes", is_statute=True)

    print(f"Buscando {len(queries)} queries...")
    with open(OUTPUT_FILE, "w") as f:
        for q in tqdm(queries):
            results = engine.search(q['text'])
            for rank, (doc_id, score) in enumerate(results):
                f.write(f"{q['id']} Q0 {doc_id} {rank+1} {score:.6f} Hybrid_InLegalBERT\n")
    print("Hecho.")

if __name__ == "__main__":
    main()