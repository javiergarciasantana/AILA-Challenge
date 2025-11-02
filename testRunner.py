from pymilvus import Collection, utility
from auxfunctions import load_embeddings
import pandas as pd
import time
import os
import re

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
        with open(file_path, 'a') as f:
          # Write model name as a header and append to CSV
          f.write(f"==={query_id}({model})===\n")
          results_df.to_csv(f, index=False, header=False)
      except FileNotFoundError:
        print(f"Directory for {file_path} not found. Creating it...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'a') as f:
          f.write(f"==={model}===\n")
          results_df.to_csv(f, index=False, header=False)
          
    def run_simple_similarity(self):
        """
        Performs a simple similarity test for each model.
        It takes the first embedding from a collection and finds its nearest neighbors
        within the same collection.
        """
        print("Running Simple Similarity Test...")
        for model in self.models:
            collection_name = f"legal_docs_{model}"
            if not utility.has_collection(collection_name):
                print(f"Collection {collection_name} not found. Skipping.")
                continue

            _, embeddings = load_embeddings(model)
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
        for model_name in self.models:
            queries_col_name = f"test_queries_{model_name}"
            docs_col_name = f"legal_docs_{model_name}"

            if not utility.has_collection(queries_col_name) or not utility.has_collection(docs_col_name):
                print(f"Required collections for model '{model_name}' not found. Skipping.")
                continue

            queries_collection = Collection(queries_col_name)
            queries_collection.load()

            docs_collection = Collection(docs_col_name)
            docs_collection.load()

            query_results = queries_collection.query(expr="id != ''", output_fields=["id", "embedding"])
            for q in query_results:
                results = docs_collection.search(
                    data=[q["embedding"]],
                    anns_field="embedding",
                    param={"nprobe": 10},
                    limit=5,
                    output_fields=["id", "chunk"]
                )

                print(f"\nQuery ID: {q['id']} (Model: {model_name})")
                for i, hit in enumerate(results[0]):
                    print(f"  {i+1}. Match: {hit.id} | Chunk: {hit.chunk} | Score: {hit.distance:.4f}")
                
                results_df = pd.DataFrame(results)
                self.csv_print(model_name, results_df, "complex_similarity_results", "Query ID:" + q['id'])
            
            queries_collection.release()
            docs_collection.release()
        
        time.sleep(3)