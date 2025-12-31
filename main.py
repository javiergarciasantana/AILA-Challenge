from pymilvus import connections, MilvusException
from auxfunctions import load_embeddings, visualize_collections, average_chars_in_textfiles
import sys
import time
from menu import Menu
from testRunner import TestRunner
from dataManager import CaseStatuteStorer, QueryStorer
from preprocessing import preprocessing_menu


def load_case_docs_and_statutes(models, kind):
    """Handler to load casedoc and statute embeddings for a chosen model."""
    print(f"\nSelect a model to load {kind} embeddings from:")
    for i, model in enumerate(models):
        print(f"  {i+1}. {model}")
    
    try:
        choice = int(input("Enter number: ")) - 1
        if 0 <= choice < len(models):
            selected_model = models[choice]
            print(f"Processing model: {selected_model}...")
            
            meta_df, embeddings = load_embeddings(selected_model, kind)
            if meta_df is not None:
                storer = CaseStatuteStorer(selected_model, kind)
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
  
  models = ['bge_base_en_v15', 'bge_small_en']

  #Connect to Milvus
  try:
    connections.connect("default", host="localhost", port="19530")
    print("✅ Successfully connected to Milvus container!")
  except MilvusException as e:
    print(f"❌ Failed to connect to Milvus: {e}")
    sys.exit(1)

  # Initialize the test runner
  test_runner = TestRunner(models)

  # Define the menu for tests
  test_menu = Menu(
      title='Please choose which test to run:',
      options=[
          ('Perform complex similarity test', test_runner.run_complex_similarity),
          ('Perform simple similarity test', test_runner.run_simple_similarity),
          ('Back', None)
      ]
  )

  # Define the main menu
  main_menu = Menu(
      title='Please choose what you wish to do:',
      options=[
          ('Run Tests', test_menu.show),
          ('Preprocessing of data', preprocessing_menu),
          ('Load Casedoc & Statutes', lambda: (load_case_docs_and_statutes(models, "casedoc"), 
                                               load_case_docs_and_statutes(models, "statute"))),
          ('Load Test Queries', lambda: load_test_queries(models)),
          ('Visualize Collections', visualize_collections),
          ('Average chars in casedocs & statutes', lambda:(average_chars_in_textfiles("./archive/Object_casedocs"),
                                                           average_chars_in_textfiles("./archive/Object_statutes"))),
          ('Exit', print("exit"))
      ]
  )
  main_menu.show()

if __name__ == "__main__" :
  main()
  connections.disconnect("default")