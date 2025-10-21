from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from auxfunctions import load_embeddings
from pick import pick
import numpy as np
import sys
import pandas as pd
import time

def ensure_index(collection_name):
    collection = Collection(collection_name)

    # Only create index if not already present
    index_info = collection.indexes
    if len(index_info) == 0:
        num_entities = collection.num_entities
        nlist = int(np.sqrt(num_entities)) if num_entities > 0 else 128

        index_params = {
            "metric_type": "IP",        # inner product similarity
            "index_type": "IVF_FLAT",   # clustered flat index
            "params": {"nlist": nlist}
        }

        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        print(f"✅ Index created for collection {collection_name}")

    collection.load()  # Load into memory for searching
    return collection


def store_cs(meta_df, embeddings, model):
  # 1. Collection creating

  _, dim = embeddings.shape

  fields = [
      FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=200),
      FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=20),
      FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=20),
      FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)  # adjust dim(embeddings.shape())
  ]

  schema = CollectionSchema(fields, description="Casedoc and statute embeddings")

  # Create the collection
  collection = Collection("legal_docs_" + model, schema)
  print("Collection ", collection.name, " created")

  # 2. Lets insert the data into the collection(into batches, message could be too big)

  batch_size = 500
  for i in range(0, len(embeddings), batch_size):
      batch_ids = meta_df["id"][i:i+batch_size].tolist()
      batch_chunks = meta_df["chunk"][i:i+batch_size].tolist()
      batch_types = meta_df["type"][i:i+batch_size].tolist() if "type" in meta_df.columns else ["unknown"] * len(batch_ids)
      batch_embeddings = embeddings[i:i+batch_size].tolist()

      collection.insert([batch_ids, batch_chunks, batch_types, batch_embeddings])


  collection.flush() # ensures data is saved and ready
  ensure_index(collection.name)
  print("Data inserted into Milvus")

def store_t_q(meta_df, embeddings, model_name):
  _, dim = embeddings.shape

  fields = [
      FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=20),
      FieldSchema(name="model", dtype=DataType.VARCHAR, max_length=20),
      FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)  # adjust dim(embeddings.shape())
  ]

  schema = CollectionSchema(fields, description="Test queries embeddings")
  # Create the collection
  collection = Collection("test_queries_" + model_name, schema)
  print("Collection ", collection.name, " created")

  ids = meta_df["id"].tolist()
  model = meta_df["model"].tolist()

  collection.insert([ids, model, embeddings.tolist()])
  collection.flush() # ensures data is saved and ready
  print("Data inserted into Milvus")
  time.sleep(5)


def simple_similarity(models):
  for model in models:

    _, embeddings = load_embeddings(model)

    collection = Collection("legal_docs_" + model)

    query_vector = embeddings[0].tolist() # -> The embedding of C1_chunk0     
    
    # Here we just do a very simple search to test similarity
    results = collection.search(
      data=[query_vector],             # must be a list of vectors
      anns_field="embedding",          # the field to search
      param={"nprobe": 10},            # how many clusters to probe
      limit=5,                         # how many nearest neighbors to return
      output_fields=["id", "type", "chunk"]     # extra metadata to return
    )
    
    # Save the top-5 most semantically similar embeddings into a CSV file
    results_data = []
    results_data.append("===" + model + "===")
    for hit in results[0]:
      results_data.append({"ID": hit.id, "Score": hit.distance})

    results_df = pd.DataFrame(results_data)

    # Append to the CSV file if it exists, otherwise create it
    with open("tests/similarity_results.csv", 'a') as f:
      results_df.to_csv(f, header=f.tell() == 0, index=False)
        
  collection.release()  # free memory after testing

  print("Results saved to tests/similarity_results.csv")
  # Wait for 5 seconds before proceeding
  time.sleep(5)

def complex_similarity(models):
  # Get all collection names
  all_collections = utility.list_collections()

  # Filter those that start with "legal_docs_"
  legal_doc_collections = [c for c in all_collections if c.startswith("legal_docs_")]




  for col_name in legal_doc_collections:
    model_name = col_name.removeprefix("legal_docs_")
    
    queries_collection = Collection("test_queries_" + model_name)
    queries_collection.load()


    # Load the legal_docs collection
    collection = Collection(col_name)
    collection.load()

    # Run similarity search for each query vector
    query_results = queries_collection.query(expr=None, output_fields=["id", "embedding"])
    for q in query_results:
        query_id = q["id"]
        query_vector = q["embedding"]

        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"nprobe": 10},
            limit=5,
            output_fields=["id", "type", "chunk"]
        )

        # Display results
        print(f"\nQuery ID: {query_id}")
        for hit in results[0]:
            print(f" - Match: {hit.id} | Score: {hit.distance:.4f}")

  
        time.sleep(5)
    

def menu_cs(models):
  while(True):
    #Choose which embeddings to use

    title = 'Please choose your preferred embedding model: '
    options = models + ['Back']

    selected_model = pick(options, title, indicator='=>', default_index=1)[0]

    if selected_model == 'Back': return

    meta_df, embeddings = load_embeddings(selected_model)

    if not utility.has_collection("legal_docs_" + selected_model):
        store_cs(meta_df, embeddings, selected_model)
        print("Collection created and data stored.")
    else:
        print("Collection already exists, skipping data insertion.")
    

def menu_q(models):
  while(True):
    title = 'Please choose what you wish to do: '
    options = ['Load test queries', 'Load all queries', 'Back']

    selected_option = pick(options, title, indicator='=>', default_index=1)[0]

    if selected_option == 'Back': return

    if selected_option == options[0]:
      for model in models:
        meta_df, embeddings = load_embeddings("queries/" + model)

        if not utility.has_collection("test_queries_" + model):
            store_t_q(meta_df, embeddings, model)
            print("Collection created and data stored.")
        else:
            print("Collection already exists, skipping data insertion.")
    else: return


def menu_test(models):
  while(True):
    title = 'Please choose what you wish to do: '
    options = ['Perform simple similarity test', 'Perform complex similarity test', 'Back']

    selected_option = pick(options, title, indicator='=>', default_index=1)[0]

    if selected_option == 'Back': return

    if selected_option == options[0]:
      print("Running Simple Similarity Test...")
      simple_similarity(models)
    else: complex_similarity(models)
  

def main():
  
  models = ['all_mpnet_base_v2', 'all_MiniLM_L6_v2', 'multi_qa_mpnet_base_dot_v1', 'all_distilroberta_v1']

  #Connect to Milvus
  connections.connect("default", host="localhost", port="19530")
  print("Succesfully connected to milvus container!")

  while(True):

    title = 'Please choose what you wish to do: '
    options = ['Load Casedoc & Statutes', 'Load queries', "Run test", 'Exit']

    selected_option = pick(options, title, indicator='=>', default_index=1)[0]

    if selected_option == options[3]:
      sys.exit()
    elif selected_option == options[0]:
      menu_cs(models)
    elif selected_option == options[1]:
      menu_q(models)
    else:
      menu_test(models)


if __name__ == "__main__" :
  main()