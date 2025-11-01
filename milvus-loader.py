from pymilvus import connections
from auxfunctions import load_embeddings, visualize_collections
import sys
import time
from menu import Menu
from testRunner import TestRunner
from dataManager import CaseStatuteStorer, QueryStorer


def load_case_docs_and_statutes(models):
    """Handler to load casedoc and statute embeddings for a chosen model."""
    print("\nSelect a model to load embeddings from:")
    for i, model in enumerate(models):
        print(f"  {i+1}. {model}")
    
    try:
        choice = int(input("Enter number: ")) - 1
        if 0 <= choice < len(models):
            selected_model = models[choice]
            print(f"Processing model: {selected_model}...")
            
            meta_df, embeddings = load_embeddings(selected_model)
            if meta_df is not None:
                storer = CaseStatuteStorer(selected_model)
                storer.store(meta_df, embeddings)
        else:
            print("Invalid choice.")
    except (ValueError, TypeError):
        print("Invalid input or failed to load embeddings. Please try again.")
    time.sleep(2)


def load_test_queries(models):
    """Handler to load test query embeddings for all models."""
    print("\nLoading test queries for all available models...")
    for model in models:
        try:
            # Queries are in a subfolder, e.g., 'export/queries/all_mpnet_base_v2'
            meta_df, embeddings = load_embeddings(f"queries/{model}")
            if meta_df is not None:
                storer = QueryStorer(model)
                storer.store_test_queries(meta_df, embeddings)
            else:
                print(f"Could not load test query embeddings for '{model}'. Skipping.")
        except FileNotFoundError:
            print(f"Embeddings file for query model '{model}' not found. Skipping.")
    print("\nFinished loading test queries.")
    time.sleep(2)


def main():
  
  models = ['all_mpnet_base_v2', 'all_MiniLM_L6_v2', 'multi_qa_mpnet_base_dot_v1', 'all_distilroberta_v1']

  #Connect to Milvus
  try:
    connections.connect("default", host="localhost", port="19530")
    print("✅ Successfully connected to Milvus container!")
  except Exception as e:
    print(f"❌ Failed to connect to Milvus: {e}")
    sys.exit(1)

  # Initialize the test runner
  test_runner = TestRunner(models)

  # Define the menu for tests
  test_menu = Menu(
      title='Please choose which test to run:',
      options=[
          ('Perform simple similarity test', test_runner.run_simple_similarity),
          ('Perform complex similarity test', test_runner.run_complex_similarity),
          ('Back', None)
      ]
  )

  # Define the main menu
  main_menu = Menu(
      title='Please choose what you wish to do:',
      options=[
          ('Load Casedoc & Statutes', lambda: load_case_docs_and_statutes(models)),
          ('Load Test Queries', lambda: load_test_queries(models)),
          ('Run Tests', test_menu.show),
          ('Visualize Collections', visualize_collections),
          ('Exit', sys.exit)
      ]
  )

  main_menu.show()
  print("Exiting program.")


if __name__ == "__main__" :
  main()