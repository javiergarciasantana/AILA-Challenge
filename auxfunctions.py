# Universidad de La Laguna
# Escuela Superior de Ingenieria y Tecnologia
# Grado en Ingenieria Informatica
# Asignatura: Trabajo Fin de Grado
# Autor: Javier Garcia Santana
# Correo: alu0101391663@ull.edu.es
# Fecha: 10/03/2025
#
# Archivo auxfuncions.py:
# Este archivo contiene funciones auxiliares para la carga, procesamiento y visualización
# de documentos legales, consultas y juicios. Estas funciones están diseñadas para trabajar
# con los datos del AILA-Challenge, facilitando la manipulación y análisis de los mismos.

# Enlaces de interes: https://www.kaggle.com/datasets/ananyapam7/legalai/data
#
# Historial de revisiones
# 10/03/2025 - Creacion (primera version) del codigo
# 27/09/2025 - Implementación de librerías para vector database embedding

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import pandas as pd
import numpy as np
import time
import os

# ==========================
# File Reading and Processing
# ==========================
def read_file(file_path):
  with open(file_path, 'r', encoding='utf-8') as file:  
    content = file.read()  
  return content

def process_query_to_dict(line):
  parts = line.strip().split("||", 1)  # Split at '||', only once
  if len(parts) == 2:  # Ensure there are exactly 2 parts
    key, value = parts
    #print(f"Key: {key}, Value: {value[0]}")  # Debugging: show the extracted key and value
    return {key: value}
  print("Line format is incorrect, returning empty dictionary.")  # Debugging: indicate incorrect format
  return {}

def process_judgment_to_dict(line, result_dict):
    parts = line.strip().split()  # Split by whitespace
    if len(parts) != 4:  # Ensure there are exactly 4 columns
        return  # Ignore incorrectly formatted lines

    key, attr1, attr2, last_col = parts  # Unpack columns

    if last_col == "1":  # Check if last column is "1"
        if key in result_dict:
            result_dict[key].append((attr1, attr2))  # Append new values
        else:
            result_dict[key] = [(attr1, attr2)]  # Create a new list


def load_objects(type, folder_path):
  docs = {}  
  if type == "casedocs":
    range_objects = range(1, 2915)
    obj_name = "C"  
  elif type == "statutes":
    range_objects = range(1, 201)
    obj_name = "S" 

  for i in range_objects:  
      file_name = f"{obj_name}{i}.txt"
      file_path = os.path.join(folder_path, file_name)  
      
      if os.path.exists(file_path):  # Check if file exists
          case_name = os.path.splitext(file_name)[0]  # Get filename without extension
          docs[case_name] = read_file(file_path)  # Read file and store in dictionary
  return docs  


def load_queries(file_path): 
  queries = {}
  with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        queries.update(process_query_to_dict(line))
      
  return queries

def load_judgments(file_path):
  judgments = {}
  with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        process_judgment_to_dict(line, judgments)
      
  return judgments

def visualize_docs(docs):
  # Convert to DataFrame for visualization
  df_docs = pd.DataFrame(docs)

  # Show first few rows (text truncated for readability)
  pd.set_option("display.max_colwidth", 100)  # show first 100 chars of text
  print(df_docs.head(10))

def all_underscores(model_name):
  return model_name.replace("-", "_")

# ==========================
# Embedding Saving and Loading
# ==========================

def save_embeddings_cs(embeddings, chunked_docs, model):
  dir_path = "export/" + model

  try:
      os.makedirs(dir_path, exist_ok=True)
  except OSError as e:
      print(f"Error: {e}")
  try:

      # 1. Save embeddings as a TSV file
      np.savetxt(dir_path + "/embeddings.tsv", np.array(embeddings), delimiter="\t")
      print("✅ Embeddings saved successfully.")

      # 2. Save metadata (text + type + id + chunk) as TSV
      metadata_df = pd.DataFrame({
          "id": [d["id"] for d in chunked_docs],
          "chunk": [d["chunk"] for d in chunked_docs],
          "type": [d["type"] for d in chunked_docs],
          "text": [d["text"][:200].replace("\n", " ") for d in chunked_docs]  # truncate for readability
      })
      metadata_df.to_csv(dir_path + "/metadata.tsv", sep="\t", index=False)
      print("✅ Metadata saved successfully.")

      return True  # indicate success

  except Exception as e:
      print(f"❌ Error while saving embeddings or metadata: {e}")
      return False  # indicate failure

def save_embeddings_q(embeddings, texts, ids, model):
  dir_path = "export/queries/" + model

  try:
      os.makedirs(dir_path, exist_ok=True)
  except OSError as e:
      print(f"Error: {e}")

  try:
     
    # Save embeddings as a TSV file
    np.savetxt(dir_path + "/embeddings.tsv", np.array(embeddings), delimiter="\t")
    print("✅ Embeddings saved successfully.")

    # Prepare metadata
    metadata_df = pd.DataFrame({
        "id": ids,
        "model": model,
        "text": texts
    })
    metadata_df.to_csv(dir_path + "/metadata.tsv", sep="\t", index=False)
    print("✅ Metadata saved successfully.")
    print(f"✅ Done with {model}!")

    return True
  
  except Exception as e:
    print(f"❌ Error while saving embeddings or metadata of queries: {e}")
    return False  # indicate failure

  
def load_embeddings(model):
  dir_path = "export/" + model
  try:
    # Load metadata
    meta_df = pd.read_csv(dir_path + "/metadata.tsv", sep="\t")

    # Load embeddings
    embeddings = np.loadtxt(dir_path + "/embeddings.tsv", delimiter="\t")

  except Exception as e:
      print(f"❌ Error while loading embeddings or metadata: {e}")
      return False  # indicate failure

  # Verify alignment
  #print(meta_df.head())
  #print(embeddings.shape)

  # Return required data
  return meta_df, embeddings

# ===========================
# Collection helper Functions
# ===========================
def ensure_index(collection_name, index_type):
    collection = Collection(collection_name)

    # Only create index if not already present
    index_info = collection.indexes
    if len(index_info) == 0:
        num_entities = collection.num_entities
        nlist = int(np.sqrt(num_entities)) if num_entities > 0 else 128

        index_params = {
            "metric_type": "IP",        # inner product similarity
            "index_type": index_type,   # clustered flat index or simple flat
            "params": {"nlist": nlist}
        }

        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        print(f"✅ Index created for collection {collection_name}")

    collection.load()  # Load into memory for searching
    return collection

def visualize_collections():
  collections = utility.list_collections()
  print("\n📦 Collections in Milvus:")
  for i, collection in enumerate(collections, start=1):
     print(f"  {i}. {collection}")
  print("\n✨ Total collections:", len(collections))
  time.sleep(5)