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
OUTPUT_FILE = "retrieval_results_agentic.txt"

# Connection Configs
MILVUS_URI = "http://localhost:19530"
OLLAMA_BASE_URL = "http://localhost:11434/v1" # Or http://host.docker.internal:11434/v1
OLLAMA_MODEL = "llama3"

# Search Params
TOP_K_RETRIEVAL = 100
FINAL_K_CASES = 10
FINAL_K_STATUTES = 10

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
        query_vecs = self.ef([query_text])
        
        # 2. Search Requests
        req_dense = AnnSearchRequest(
            data=[query_vecs["dense"][0]], anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}}, limit=TOP_K_RETRIEVAL
        )
        req_sparse = AnnSearchRequest(
            data=[query_vecs["sparse"][0]], anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}, limit=TOP_K_RETRIEVAL
        )
        
        # 3. RRF Fusion (Combines the two results)
        res = self.client.hybrid_search(
            collection_name, reqs=[req_dense, req_sparse],
            ranker=RRFRanker(k=60), limit=TOP_K_RETRIEVAL, output_fields=["doc_name"]
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

def main():
    # 1. Load Data
    cases = load_files(PATH_CASES, "C")
    statutes = load_files(PATH_STATUTES, "S")
    queries = []
    with open(PATH_QUERIES, errors='ignore') as f:
        for l in f: 
            if "||" in l: queries.append({"id": l.split("||")[0], "text": l.split("||")[1].strip()})

    # 2. Initialize Systems
    agent = AgenticLayer()
    db = RetrievalSystem()
    
    # 3. Index (Auto-skips if cached)
    db.process_and_index("aila_cases", cases)
    db.process_and_index("aila_statutes", statutes)

    print(f"\n🚀 Running Agentic RAG on {len(queries)} queries...")
    with open(OUTPUT_FILE, "w") as f:
        for q in tqdm(queries):
            # A. Reasoning Layer (Facts -> Legal Terms)
            enhanced_query = agent.expand_query_with_reasoning(q['text'])
            
            # B. Parallel Hybrid Retrieval
            cands_c = db.search_hybrid("aila_cases", enhanced_query)
            cands_s = db.search_hybrid("aila_statutes", enhanced_query)
            
            # C. Reranking
            ranked_c = db.rerank_sliding_window(enhanced_query, cands_c, FINAL_K_CASES)
            ranked_s = db.rerank_sliding_window(enhanced_query, cands_s, FINAL_K_STATUTES)
            
            # D. Merge
            final_list = ranked_c + ranked_s
            final_list.sort(key=lambda x: x[1], reverse=True) # Final mix
            
            for rank, (doc_id, score) in enumerate(final_list):
                f.write(f"{q['id']} Q0 {doc_id} {rank+1} {score:.4f} AgenticHybrid\n")

    print(f"✅ Done. Results at {OUTPUT_FILE}")

if __name__ == "__main__":
    import glob
    main()