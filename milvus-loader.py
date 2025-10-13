from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from auxfunctions import load_embeddings
from pick import pick
import numpy as np
import pandas as pd

def store_data(meta_df, embeddings, model):
  # 1. Collection creating

  row_num, dim = embeddings.shape

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
  print("Data inserted into Milvus")


def main():
  #Connect to Milvus
  connections.connect("default", host="localhost", port="19530")
  print("Succesfully connected to milvus container!")

  #Choose which embeddings to use

  title = 'Please choose your preferred embedding model: '
  options = ['all_mpnet_base_v2', 'all_MiniLM_L6_v2', 'multi_qa_mpnet_base_dot_v1', 'all_distilroberta_v1']

  selected_model, index = pick(options, title, indicator='=>', default_index=1)

  meta_df, embeddings = load_embeddings(selected_model)

  if not utility.has_collection("legal_docs_" + selected_model):
      store_data(meta_df, embeddings, selected_model)
      print("Collection created and data stored.")
  else:
      print("Collection already exists, skipping data insertion.")
    
  collection = Collection("legal_docs_" + selected_model)
  index_params = {
    "metric_type": "IP",        # inner product similarity (the higher the better)
    "index_type": "IVF_FLAT",   # a clustering index type (medium datasets)
    "params": {"nlist": int(np.sqrt(collection.num_entities))}    # number of clusters to create, good heuristic -> nlist = int(sqrt(num_vectors))
  }

  collection.create_index(field_name="embedding", index_params=index_params)
  collection.load() # loads the indexed data into memory -> ready to search
  
  query_vector = embeddings[0].tolist() # -> The embedding of C1_chunk0     
  
  # Here we just do a very simple search to test similarity
  results = collection.search(
    data=[query_vector],             # must be a list of vectors
    anns_field="embedding",          # the field to search
    param={"nprobe": 10},            # how many clusters to probe
    limit=5,                         # how many nearest neighbors to return
    output_fields=["id", "type", "chunk"]     # extra metadata to return
  )
  
  print("\nTest Results\n")
  # The top-5 most semantically similar embeddings are these, with their similarity scores.
  for hit in results[0]:
    print(f"ID: {hit.id} | Score: {hit.distance:.4f}")

if __name__ == "__main__" :
  main()