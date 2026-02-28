import os
import re
import glob
import unicodedata
import numpy as np
from tqdm import tqdm
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

# --- CONFIGURACIÓN ---
BASE_DIR = "/Users/javiersantana/INF_2024_2025/Trabajo Fin de Grado/AILA-Challenge/archive"
PATH_CASES = os.path.join(BASE_DIR, "Object_casedocs")
PATH_STATUTES = os.path.join(BASE_DIR, "Object_statutes")
PATH_QUERIES = os.path.join(BASE_DIR, "Query_doc.txt")

# ARCHIVOS DE SALIDA Y EVALUACIÓN
os.makedirs("test_results/v_dense", exist_ok=True)
TREC_FILE = "test_results/v_dense/trec_rankings.txt"
OUTPUT_METRICS = "test_results/v_dense/eval_metrics.txt"
QRELS_FILES = [
    os.path.join(BASE_DIR, "relevance_judgments_statutes.txt"),
    os.path.join(BASE_DIR, "relevance_judgments_priorcases.txt")
]

MILVUS_URI = "http://localhost:19530"
EMBEDDING_MODEL = "./models/InLegalBERT-AILA-Tuned" 

# PARÁMETROS
CHUNK_SIZE = 300 # Palabras (Asegura no pasar de 512 tokens)
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 100
K_METRICS = 60

class TextProcessor:
    @staticmethod
    def super_clean(text):
        """
        Limpiador estándar NLP robusto.
        Mantiene la puntuación porque BERT la necesita para el contexto.
        """
        # 1. Normalizar caracteres Unicode (elimina acentos raros, espacios invisibles)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        # 2. Eliminar URLs o links que ensucian el embedding
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # 3. Eliminar saltos de línea múltiples y tabulaciones
        text = re.sub(r'[\r\n\t]+', ' ', text)
        # 4. Colapsar espacios múltiples en uno solo
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def chunk_text(text):
        words = text.split()
        if not words: return []
        chunks = []
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = " ".join(words[i:i + CHUNK_SIZE])
            chunks.append(chunk)
        return chunks

class DenseSearchSystem:
    def __init__(self):
        print(f"⏳ Cargando Modelo Denso: {EMBEDDING_MODEL}...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL) 
        self.dim = 768
        self.client = MilvusClient(uri=MILVUS_URI)

    def index_collection(self, documents, collection_name):
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
        
        # 1. Definir el esquema de la colección
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="doc_name", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim)
        
        # 2. Configurar los parámetros del índice HNSW
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector", 
            index_type="HNSW", 
            metric_type="COSINE", 
            params={"M": 16, "efConstruction": 200}
        )

        # 3. Crear la colección aplicando el esquema y el índice
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )

        all_chunks = []
        for doc in tqdm(documents, desc=f"Procesando {collection_name}"):
            clean_text = TextProcessor.super_clean(doc['text'])
            chunks = TextProcessor.chunk_text(clean_text)
            for chunk in chunks:
                all_chunks.append({"doc_name": doc['doc_name'], "text": chunk})

        batch_size = 32
        for i in tqdm(range(0, len(all_chunks), batch_size), desc=f"Vectorizando {collection_name}"):
            batch = all_chunks[i : i + batch_size]
            texts = [x['text'] for x in batch]
            vectors = self.encoder.encode(texts, normalize_embeddings=True)
            
            payload = [{"vector": vec, "doc_name": batch[idx]['doc_name']} for idx, vec in enumerate(vectors)]
            self.client.insert(collection_name=collection_name, data=payload)
            
        # 4. Cargar la colección en memoria (Vital para poder buscar rápidamente después)
        print(f"⏳ Cargando índice en memoria para {collection_name}...")
        self.client.load_collection(collection_name)
        

    def search(self, query_text):
        # Limpieza de la consulta
        q_clean = TextProcessor.super_clean(query_text)
        q_vector = self.encoder.encode([q_clean], normalize_embeddings=True)
        
        # Búsqueda en Casos
        res_cases = self.client.search(
            collection_name="aila_cases_v3", data=q_vector, limit=TOP_K_RETRIEVAL, output_fields=["doc_name"]
        )
        
        # Búsqueda en Estatutos
        res_statutes = self.client.search(
            collection_name="aila_statutes_v3", data=q_vector, limit=TOP_K_RETRIEVAL, output_fields=["doc_name"]
        )

        # Unir y ordenar por distancia Coseno (Mayor es mejor en Milvus COSINE)
        combined_results = []
        seen = set()
        
        # Función auxiliar para procesar hits y evitar duplicados (mismo documento, distinto chunk)
        def process_hits(hits):
            for hit in hits[0]:
                doc_name = hit['entity']['doc_name']
                score = hit['distance']
                if doc_name not in seen:
                    combined_results.append((doc_name, score))
                    seen.add(doc_name)

        process_hits(res_cases)
        process_hits(res_statutes)
        
        # Ordenar globalmente por similitud semántica
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:TOP_K_RETRIEVAL]

# --- FUNCIONES DE EVALUACIÓN ---
def load_docs(path, prefix):
    docs = []
    for f in glob.glob(os.path.join(path, f"{prefix}*.txt")):
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
    qrels = {}
    for filepath in filepaths:
        if not os.path.exists(filepath): continue
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4 and int(parts[3]) > 0:
                    qrels.setdefault(parts[0], set()).add(parts[2])
    return qrels

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = [doc for doc in retrieved_k if doc in relevant]
    return len(hits) / k if k else 0.0

def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = [doc for doc in retrieved_k if doc in relevant]
    return len(hits) / len(relevant) if relevant else 0.0

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
    print("🚀 Iniciando Sistema de Recuperación Puramente Denso...")
    
    # 1. Cargar Datos
    cases = load_docs(PATH_CASES, "C")
    statutes = load_docs(PATH_STATUTES, "S")
    queries = load_queries(PATH_QUERIES)
    
    # 2. Indexación (Solo Vectores)
    system = DenseSearchSystem()
    if cases: system.index_collection(cases, "aila_cases_v3")
    if statutes: system.index_collection(statutes, "aila_statutes_v3")
    
    # 3. Búsqueda y Generación de Archivo TREC
    print(f"\n🔍 Procesando {len(queries)} consultas...")
    rankings_dict = {}
    
    with open(TREC_FILE, "w") as f:
        for q in tqdm(queries, desc="Buscando"):
            results = system.search(q['text'])
            rankings_dict[q['id']] = [doc_id for doc_id, score in results]
            
            for rank, (doc_id, score) in enumerate(results):
                f.write(f"{q['id']} Q0 {doc_id} {rank+1} {score:.4f} Dense_InLegalBERT\n")
                
    print(f"✅ Resultados guardados en {TREC_FILE}")
    
    # 4. Evaluación Automática
    print("\n📊 Calculando Métricas de Evaluación...")
    qrels = load_qrels(QRELS_FILES)
    total_p, total_r, total_ap, n = 0, 0, 0, 0

    with open(OUTPUT_METRICS, "w") as fout:
        for qid, retrieved in rankings_dict.items():
            relevant = qrels.get(qid, set())
            if relevant:
                p = precision_at_k(retrieved, relevant, K_METRICS)
                r = recall_at_k(retrieved, relevant, K_METRICS)
                ap = average_precision(retrieved, relevant, K_METRICS)
                
                n += 1
                total_p += p
                total_r += r
                total_ap += ap
                fout.write(f"QUERY: {qid:<10} | P@{K_METRICS}: {p:.4f} | R@{K_METRICS}: {r:.4f} | MAP@{K_METRICS}: {ap:.4f}\n")
        
        if n > 0:
            mean_p, mean_r, mean_ap = total_p / n, total_r / n, total_ap / n
            report = (
                f"\n{'='*50}\n"
                f"📊 REPORTE DE EVALUACIÓN GLOBAL (Solo Vectores Densos)\n"
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