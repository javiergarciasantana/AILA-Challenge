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

import pandas as pd
import numpy as np
import os


def read_file(file_path):
  with open(file_path, 'r', encoding='utf-8') as file:  
    content = file.read()  
  return content


def process_query_to_dict(line):
    parts = line.strip().split("||", 1)  # Split at '||', only once
    if len(parts) == 2:  # Ensure there are exactly 2 parts
        key, value = parts
        return {key: value}
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
        line = file.readline() 
        queries.update(process_query_to_dict(line))
      
  return queries

def load_judgments(file_path):
  judgments = {}
  with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = file.readline() 
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

def save_embeddings(embeddings, chunked_docs, model):
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
  print(embeddings.shape)

  # Return required data
  return meta_df, embeddings