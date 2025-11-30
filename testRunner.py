from pymilvus import Collection, utility
from auxfunctions import load_embeddings, is_expected, count_expected, aggregate_by_case
import pandas as pd
import time
import os
import re
from pathlib import Path
from datetime import datetime


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
      If the file does not exist, it creates a new one.
      Args:
        model (str): The name of the model.
        results_df (pd.DataFrame): The DataFrame containing the results to be written.
      """
      file_path = f"tests/{filename}.csv"
      
      try:
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
          # Write model name as a header and append to CSV
          f.write(f"=== {query_id} ({model}) ===\n")
          results_df.to_csv(f, index=False, header=True)
      except FileNotFoundError:
        print(f"Directory for {file_path} not found. Creating it...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
          f.write(f"=== {query_id} ({model}) ===\n")
          results_df.to_csv(f, index=False, header=True)
    
    def xlsx_print(self, all_rows_statute, all_rows_casedoc):
      # Export to Excel with proper sheets
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
        It takes the first embedding from a collection and finds its nearest neighbors
        within the same collection.
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
        Performs a complex similarity test.
        It loads embeddings from a dedicated 'test_queries' collection for each model
        and searches for neighbors in the corresponding 'legal_docs' collection.
        """
        print("Running Complex Similarity Test...")
        all_rows_statute = []
        all_rows_casedoc = []

        for model_name in self.models:
          #model_name = "multi_qa_mpnet_base_dot_v1"
          queries_col_name = f"test_queries_{model_name}"
          satute_docs_col_name = f"statute_{model_name}"
          casedoc_docs_col_name = f"casedoc_{model_name}"

          if (
              not utility.has_collection(queries_col_name) or 
              not utility.has_collection(satute_docs_col_name) or 
              not utility.has_collection(casedoc_docs_col_name)
          ):
              print(f"Required collections for model '{model_name}' not found. Skipping.")
              continue

          queries_collection = Collection(queries_col_name)
          queries_collection.load()

          statute_collection = Collection(satute_docs_col_name)
          statute_collection.load()

          casedoc_collection = Collection(casedoc_docs_col_name)
          casedoc_collection.load()

          metric = "COSINE" if model_name.startswith('bge') else "L2"
          search_params = {
            "metric_type": metric,
            "offset": 0,
            "ignore_growing": False,
            "params": {"nprobe": 50}
          }
          query_results = queries_collection.query(expr="id != ''", output_fields=["id", "embedding"])
          for q in query_results:
              statute_precision = 0
              casedoc_precision = 0
              statute_count = 0
              casedoc_count = 0

              results_s = statute_collection.search(
                  data=[q["embedding"]],
                  anns_field="embedding",
                  param=search_params,
                  limit=20,
                  output_fields=["id", "chunk"]
              )

              results_c = casedoc_collection.search(
                  data=[q["embedding"]],
                  anns_field="embedding",
                  param=search_params,
                  limit=30,
                  output_fields=["id", "chunk"]
              )
              rows_s = []
              for i, hit in enumerate(results_s[0]):
                  statute_count = (count_expected("statute", q["id"]))
                  if is_expected("statute", q["id"], hit.id):
                      statute_precision += 1
                      flag = "✧"
                  else : flag = ''
                  row = {
                      "query_id": q["id"],
                      "model": model_name,
                      "target": "statute",
                      "rank": i + 1,
                      "id": str(hit.id) + flag,
                      "chunk": (hit.entity.get("chunk") if hasattr(hit, "entity") else None),
                      "score": float(hit.distance),
                  }
                  rows_s.append(row)

              # --- Process Casedoc Results with a for loop ---
              rows_c = []
              for i, hit in enumerate(results_c[0]):
                  casedoc_count = (count_expected("casedoc", q["id"]))
                  if is_expected("casedoc", q["id"], hit.id):
                      casedoc_precision += 1
                      flag = "✧"
                  else : flag = ''
                  row = {
                      "query_id": q["id"],
                      "model": model_name,
                      "target": "casedoc",
                      "rank": i + 1,
                      "id": str(hit.id) + flag,
                      "chunk": (hit.entity.get("chunk") if hasattr(hit, "entity") else None),
                      "score": float(hit.distance),
                  }
                  rows_c.append(row)
             
              statute_precision, casedoc_precision = statute_precision / statute_count, casedoc_precision / casedoc_count

              if rows_s:
                  divider_s = {"query_id": f"--- Query: {q['id']}, Precision: {statute_precision} (Model: {model_name}) ---"}
                  all_rows_statute.append(divider_s)
                  all_rows_statute.extend(rows_s)
                  all_rows_statute.append({}) # Add a blank row for spacing

              if rows_c:
                  divider_c = {"query_id": f"--- Query: {q['id']}, Precision: {casedoc_precision} (Model: {model_name}) ---"}
                  all_rows_casedoc.append(divider_c)
                  all_rows_casedoc.extend(rows_c)
                  all_rows_casedoc.append({})

              agg_rows_c = aggregate_by_case(rows_c)
              # Optional console preview
              relevant_ids = set()
              with open("./archive/relevance_judgments_priorcases.txt") as f:
                  for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4 and parts[0] == q["id"] and parts[3] == "1":
                        relevant_ids.add(parts[2])

              print(f"\nQuery ID: {q['id']} (Model: {model_name}) (Precision:{statute_precision}) - Statutes")
              print(pd.DataFrame(rows_s)[["rank","id","chunk","score"]])
              print(f"\nQuery ID: {q['id']} (Model: {model_name}) (Precision:{casedoc_precision}) - CaseDocs")
              print(pd.DataFrame(agg_rows_c)[["rank","id","chunk","score"]])

              print("Relevant casedoc IDs for", q["id"], ":", relevant_ids)
              print("Top retrieved casedoc IDs:", [row["id"].split()[0] for row in agg_rows_c[:10]])
          
          queries_collection.release()
          statute_collection.release()
          casedoc_collection.release()

        self.xlsx_print(all_rows_statute, all_rows_casedoc)
        time.sleep(1)



    def run_bge_small_queries_on_bge_base_en_casedocs(self):
      """
      Uses the 'bge-small-en' test queries to search in the 'casedoc_bge-base-en' collection.
      Prints and saves the results.
      """
      queries_col_name = "test_queries_bge_base_en_v15"
      casedoc_col_name = "casedoc_bge_base_en_v15"

      casedoc_precision = 0

      if not (utility.has_collection(queries_col_name) and utility.has_collection(casedoc_col_name)):
        print(f"Required collections '{queries_col_name}' or '{casedoc_col_name}' not found.")
        return

      queries_collection = Collection(queries_col_name)
      queries_collection.load()
      casedoc_collection = Collection(casedoc_col_name)
      casedoc_collection.load()

      search_params = {
        "metric_type": "COSINE",
        "offset": 0,
        "ignore_growing": False,
        "params": {"nprobe": 50}
      }

      query_results = queries_collection.query(expr="id != ''", output_fields=["id", "embedding"])

      for q in query_results:
        results = casedoc_collection.search(
          data=[q["embedding"]],
          anns_field="embedding",
          param=search_params,
          limit=20,
          output_fields=["id", "chunk"]
        )
        rows = []
        for i, hit in enumerate(results[0]):
          casedoc_count = (count_expected("casedoc", q["id"]))
          if is_expected("casedoc", q["id"], hit.id):
              casedoc_precision += 1
              flag = "✧"
          else : flag = ''
          row = {
            "query_id": q["id"],
            "model": "bge-base-en",
            "target": "casedoc",
            "rank": i + 1,
            "id": str(hit.id) + flag,
            "chunk": (hit.entity.get("chunk") if hasattr(hit, "entity") else None),
            "score": float(hit.distance),
          }
          rows.append(row)

        casedoc_precision = casedoc_precision / casedoc_count

        print(f"\nQuery ID: {q['id']} Precision: {casedoc_precision}  ({casedoc_col_name[8:]})")
        print(pd.DataFrame(rows)[["rank", "id", "chunk", "score"]])
        relevant_ids = set()
        with open("./archive/relevance_judgments_priorcases.txt") as f:
            for line in f:
              parts = line.strip().split()
              if len(parts) == 4 and parts[0] == q["id"] and parts[3] == "1":
                  relevant_ids.add(parts[2])
        
        print("Relevant casedoc IDs for", q["id"], ":", relevant_ids)

      queries_collection.release()
      casedoc_collection.release()
