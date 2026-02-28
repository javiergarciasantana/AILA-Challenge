import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from pymilvus import Collection, utility
from auxfunctions import (
    load_embeddings,
    is_expected,
    count_expected,
    aggregate_by_case,
    rerank_with_cross_encoder
)

# --- Add these imports for metrics ---
from tqdm import tqdm

# --- Add your metrics helpers here or import them ---
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

def calculate_metrics(retrieved_docs, relevant_docs, k):
    """Calcula Precision@K, Recall@K y Average Precision@K"""
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = [doc for doc in retrieved_k if doc in relevant_docs]
    precision = len(relevant_retrieved) / k if k > 0 else 0.0
    recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0.0
    ap = 0.0
    hits = 0
    for i, doc in enumerate(retrieved_k):
        if doc in relevant_docs:
            hits += 1
            ap += hits / (i + 1)
    ap = ap / len(relevant_docs) if relevant_docs else 0.0
    return precision, recall, ap

# --- Output file paths and params ---
TREC_OUTPUT_FILE = "../test_results/v2/trec_rankings.txt"
METRICS_OUTPUT_FILE = "../test_results/v2/eval_metrics.txt"
K_METRICS = 60  # Set this to FINAL_K_CASES + FINAL_K_STATUTES or as needed
PATH_QRELS_STATUTES = "../archive/relevance_judgments_statutes.txt"
PATH_QRELS_CASES = "../archive/relevance_judgments_priorcases.txt"

class TestRunner:
    """A class to encapsulate Milvus similarity testing logic."""

    def __init__(self, models):
        self.models = models

    def csv_print(self, model, results_df, filename, query_id=''):
        file_path = f"tests/{filename}.csv"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        mode = 'a'
        header = True
        try:
            with open(file_path, mode, newline='', encoding='utf-8') as f:
                f.write(f"=== {query_id} ({model}) ===\n")
                results_df.to_csv(f, index=False, header=header)
        except Exception as e:
            print(f"Error writing to CSV: {e}")

    def xlsx_print(self, all_rows_statute, all_rows_casedoc):
        out_dir = Path("tests")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx_path = out_dir / f"complex_similarity_results_{ts}.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            if all_rows_statute:
                pd.DataFrame(all_rows_statute).to_excel(writer, sheet_name="Statutes", index=False)
            if all_rows_casedoc:
                pd.DataFrame(all_rows_casedoc).to_excel(writer, sheet_name="CaseDocs", index=False)

    def run_simple_similarity(self):
        print("Running Simple Similarity Test...")
        for model in self.models:
            collection_name = f"casedoc_{model}"
            if not utility.has_collection(collection_name):
                print(f"Collection {collection_name} not found. Skipping.")
                continue
            _, embeddings = load_embeddings(model, "casedoc")
            collection = Collection(collection_name)
            collection.load()
            query_vector = embeddings[0].tolist()
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param={"nprobe": 10},
                limit=5,
                output_fields=["id", "type", "chunk"]
            )
            results_data = [{"ID": hit.id, "Score": hit.distance} for hit in results[0]]
            results_df = pd.DataFrame(results_data)
            print(f"\n--- Results for {model} ---")
            print(results_df)
            self.csv_print(model, results_df, "simple_similarity_results")
            collection.release()
        print("\nResults saved to tests/simple_similarity_results.csv")
        time.sleep(3)

    def run_complex_similarity(self):
        """
        Performs a complex similarity test and exports TREC and metrics files.
        """
        print("Running Complex Similarity Test...")

        # --- Load ground truth for metrics ---
        qrels = load_qrels([PATH_QRELS_STATUTES, PATH_QRELS_CASES])

        # --- Prepare output files ---
        os.makedirs(os.path.dirname(TREC_OUTPUT_FILE), exist_ok=True)
        with open(TREC_OUTPUT_FILE, "w") as f: f.write("")
        with open(METRICS_OUTPUT_FILE, "w") as f: f.write("")

        # --- For metrics aggregation ---
        total_precision = 0.0
        total_recall = 0.0
        total_ap = 0.0
        queries_evaluated = 0

        # --- Iterate over models and queries ---
        # for model_name in self.models:
        queries_col_name = f"queries_bge_small_en"
        statute_col_name = f"statute_bge_small_en"
        casedoc_col_name = f"casedoc_bge_small_en"

        # if not all(utility.has_collection(name) for name in [queries_col_name, statute_col_name, casedoc_col_name]):
        #     print(f"Required collections for model '{model_name}' not found. Skipping.")
        #     continue

        queries_collection = Collection(queries_col_name)
        statute_collection = Collection(statute_col_name)
        casedoc_collection = Collection(casedoc_col_name)
        queries_collection.load()
        statute_collection.load()
        casedoc_collection.load()

        # metric = "COSINE" if model_name.startswith('bge') else "L2"
        search_params = {
            "metric_type": "COSINE",
            "offset": 0,
            "ignore_growing": False,
            "params": {"nprobe": 10}
        }

        query_results = queries_collection.query(expr="id != ''", output_fields=["id", "embedding", "text"])

        for q in tqdm(query_results, desc=f"Model bge_small_en"):
            q_id = q["id"]
            q_text = q["text"]

            # --- Statute Retrieval ---
            results_s = statute_collection.search(
                data=[q["embedding"]],
                anns_field="embedding",
                param=search_params,
                limit=K_METRICS // 2,
                output_fields=["id", "chunk", "text"]
            )
            # --- CaseDoc Retrieval ---
            results_c = casedoc_collection.search(
                data=[q["embedding"]],
                anns_field="embedding",
                param=search_params,
                limit=K_METRICS // 2,
                output_fields=["id", "chunk", "text"]
            )

            # --- Merge and rerank (if needed) ---
            final = []
            for i, hit in enumerate(results_c[0]):
                final.append((str(hit.id), float(hit.distance)))
            for i, hit in enumerate(results_s[0]):
                final.append((str(hit.id), float(hit.distance)))
            final.sort(key=lambda x: x[1], reverse=True)  # or by score, depending on your logic

            # --- Write TREC output ---
            with open(TREC_OUTPUT_FILE, "a") as f:
                for rank, (doc_id, score) in enumerate(final):
                    f.write(f"{q_id} Q0 {doc_id} {rank+1} {score:.4f} AgenticHybrid\n")

            # --- Metrics ---
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

        queries_collection.release()
        statute_collection.release()
        casedoc_collection.release()

        # --- Global report ---
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
        time.sleep(1)