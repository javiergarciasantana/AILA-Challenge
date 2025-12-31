import os
import time
import re
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

class TestRunner:
    """A class to encapsulate Milvus similarity testing logic."""

    def __init__(self, models):
        """
        Initializes the TestRunner.
        Args:
            models (list): A list of model names to run tests against.
        """
        self.models = models

    def csv_print(self, model, results_df, filename, query_id=''):
        """
        Appends the similarity results to a CSV file with the model name as a header.
        """
        file_path = f"tests/{filename}.csv"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        mode = 'a'
        header = True
        # If file exists, we might not need header, but your logic writes it every time with a separator.
        
        try:
            with open(file_path, mode, newline='', encoding='utf-8') as f:
                f.write(f"=== {query_id} ({model}) ===\n")
                results_df.to_csv(f, index=False, header=header)
        except Exception as e:
            print(f"Error writing to CSV: {e}")

    def xlsx_print(self, all_rows_statute, all_rows_casedoc):
        """Export results to Excel with proper sheets."""
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
        """
        Performs a simple similarity test for each model.
        """
        print("Running Simple Similarity Test...")
        for model in self.models:
            collection_name = f"casedoc_{model}"
            if not utility.has_collection(collection_name):
                print(f"Collection {collection_name} not found. Skipping.")
                continue

            _, embeddings = load_embeddings(model, "casedoc")
            collection = Collection(collection_name)
            collection.load()

            query_vector = embeddings[0].tolist()  # The embedding of the first chunk

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
        Performs a complex similarity test with deduplication and reranking.
        """
        print("Running Complex Similarity Test...")
        all_rows_statute = []
        all_rows_casedoc = []

        for model_name in self.models:
            queries_col_name = f"test_queries_{model_name}"
            statute_col_name = f"statute_{model_name}"
            casedoc_col_name = f"casedoc_{model_name}"

            # Check existence of all required collections
            if not all(utility.has_collection(name) for name in [queries_col_name, statute_col_name, casedoc_col_name]):
                print(f"Required collections for model '{model_name}' not found. Skipping.")
                continue

            # Load collections
            queries_collection = Collection(queries_col_name)
            statute_collection = Collection(statute_col_name)
            casedoc_collection = Collection(casedoc_col_name)
            
            queries_collection.load()
            statute_collection.load()
            casedoc_collection.load()

            metric = "COSINE" if model_name.startswith('bge') else "L2"
            search_params = {
                "metric_type": metric,
                "offset": 0,
                "ignore_growing": False,
                "params": {"nprobe": 10}
            }

            # Fetch all queries
            query_results = queries_collection.query(expr="id != ''", output_fields=["id", "embedding", "text"])

            for q in query_results:
                # --- 1. Statute Retrieval ---
                results_s = statute_collection.search(
                    data=[q["embedding"]],
                    anns_field="embedding",
                    param=search_params,
                    limit=20,
                    output_fields=["id", "chunk", "text"]
                )
                
                # Pre-fetch expected counts to avoid repeated calls in loop
                statute_expected_count = count_expected("statute", q["id"])
                casedoc_expected_count = count_expected("casedoc", q["id"])
                
                statute_hits = 0
                rows_s = []
                
                for i, hit in enumerate(results_s[0]):
                    is_hit = is_expected("statute", q["id"], hit.id)
                    if is_hit:
                        statute_hits += 1
                    
                    flag = "✧" if is_hit else ""
                    row = {
                        "query_id": q["id"],
                        "model": model_name,
                        "target": "statute",
                        "rank": i + 1,
                        "id": str(hit.id) + flag,
                        "chunk": hit.entity.get("chunk"),
                        "text": hit.entity.get("text")[38:] if hit.entity.get("text") else None,
                        "score": float(hit.distance),
                    }
                    rows_s.append(row)

                # --- 2. CaseDoc Retrieval (High Recall Phase) ---
                # Search limit increased to 2000 to overcome chunk saturation
                results_c = casedoc_collection.search(
                    data=[q["embedding"]],
                    anns_field="embedding",
                    param=search_params,
                    limit=2000, 
                    output_fields=["id", "chunk", "text"]
                )

                casedoc_hits = 0
                rows_c_raw = []
                
                for i, hit in enumerate(results_c[0]):
                    is_hit = is_expected("casedoc", q["id"], hit.id)
                    if is_hit:
                        casedoc_hits += 1
                    
                    flag = "✧" if is_hit else ""
                    row = {
                        "query_id": q["id"],
                        "model": model_name,
                        "target": "casedoc",
                        "rank": i + 1,
                        "id": str(hit.id) + flag,
                        "chunk": hit.entity.get("chunk"),
                        "text": hit.entity.get("text")[35:] if hit.entity.get("text") else None,
                        "score": float(hit.distance),
                    }
                    rows_c_raw.append(row)

                # Calculate Precision (on retrieval set)
                statute_precision = statute_hits / statute_expected_count if statute_expected_count > 0 else 0
                casedoc_precision = casedoc_hits / casedoc_expected_count if casedoc_expected_count > 0 else 0

                # --- 3. Deduplication Logic ---
                unique_cases = {}
                for row in rows_c_raw:
                    # Assumes ID format is "C123" or "C123✧". 
                    # If using flags, ensure splitting or keying handles them consistently.
                    case_id = row['id'].split()[0] 
                    
                    # Keep first occurrence (Milvus returns sorted by score, so first is best)
                    if case_id not in unique_cases:
                        unique_cases[case_id] = row
                
                # Slice Top 100 Unique Candidates
                distinct_candidates = list(unique_cases.values())[:100]

                # --- 4. Reranking ---
                # Truncate query to 1000 chars to fit CrossEncoder context
                truncated_query = q["text"][:1000]
                rows_s_reranked = rerank_with_cross_encoder(truncated_query, rows_s)
                rows_c_reranked = rerank_with_cross_encoder(truncated_query, distinct_candidates)

                # --- 5. Formatting for Output ---
                def limit_text(row):
                    txt = row.get("text", "")
                    return (txt[:20] + "...") if txt and len(txt) > 20 else txt

                rows_s_display = [{**row, "text": limit_text(row)} for row in rows_s_reranked]
                rows_c_display = [{**row, "text": limit_text(row)} for row in rows_c_reranked]

                if rows_s_display:
                    header = {"query_id": f"--- Query: {q['id']}, Precision: {statute_precision:.2f} (Model: {model_name}) ---"}
                    all_rows_statute.append(header)
                    all_rows_statute.extend(rows_s_display)
                    all_rows_statute.append({}) 

                if rows_c_display:
                    header = {"query_id": f"--- Query: {q['id']}, Precision: {casedoc_precision:.2f} (Model: {model_name}) ---"}
                    all_rows_casedoc.append(header)
                    all_rows_casedoc.extend(rows_c_display)
                    all_rows_casedoc.append({})

                # --- Console Output ---
                relevant_ids = set()
                try:
                    with open("./archive/relevance_judgments_priorcases.txt") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 4 and parts[0] == q["id"] and parts[3] == "1":
                                relevant_ids.add(parts[2])
                except FileNotFoundError:
                    pass # Skip if file not found

                print(f"\nQuery ID: {q['id']} (Model: {model_name}) - Statutes")
                print(pd.DataFrame(rows_s_reranked)[["rank", "id", "chunk", "score"]])
                
                print(f"\nQuery ID: {q['id']} (Model: {model_name}) - CaseDocs (Top Unique Reranked)")
                print(pd.DataFrame(rows_c_reranked)[["rank", "id", "chunk", "score"]])

                print("Relevant casedoc IDs for", q["id"], ":", relevant_ids)
                # Show top 10 reranked IDs for quick check
                print("Top retrieved casedoc IDs:", [row["id"].split()[0] for row in rows_c_reranked[:10]])

            queries_collection.release()
            statute_collection.release()
            casedoc_collection.release()

        # Uncomment to save Excel at end of run
        # self.xlsx_print(all_rows_statute, all_rows_casedoc)
        time.sleep(1)