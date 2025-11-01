from pymilvus import Collection, utility
from auxfunctions import load_embeddings
import pandas as pd
import time

class TestRunner:
    """A class to encapsulate Milvus similarity testing logic."""
    def __init__(self, models):
        """
        Initializes the TestRunner.
        Args:
            models (list): A list of model names to run tests against.
        """
        self.models = models

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

            with open("tests/similarity_results.csv", 'a') as f:
                # Prepend model name and write to CSV
                f.write(f"\n==={model}===\n")
                results_df.to_csv(f, header=f.tell() < 20, index=False)
            
            collection.release()

        print("\nResults saved to tests/similarity_results.csv")
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

            query_results = queries_collection.query(expr=None, output_fields=["id", "embedding"])
            for q in query_results:
                results = docs_collection.search(
                    data=[q["embedding"]],
                    anns_field="embedding",
                    param={"nprobe": 10},
                    limit=5,
                    output_fields=["id"]
                )

                print(f"\nQuery ID: {q['id']} (Model: {model_name})")
                for i, hit in enumerate(results[0]):
                    print(f"  {i+1}. Match: {hit.id} | Score: {hit.distance:.4f}")
            
            queries_collection.release()
            docs_collection.release()
        
        time.sleep(3)