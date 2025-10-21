# Universidad de La Laguna
# Escuela Superior de Ingenieria y Tecnologia
# Grado en Ingenieria Informatica
# Asignatura: Trabajo Fin de Grado
# Autor: Javier Garcia Santana
# Correo: alu0101391663@ull.edu.es
# Fecha: 10/03/2025
#
# Archivo preprocessing.py: Este programa se utiliza para leer y cargar documentos legales desde archivos de texto. 
# Proporciona funciones para leer el contenido de un archivo y cargar múltiples archivos 
# en un diccionario, categorizados como "casedocs" o "statutes". Además, el programa divide los documentos en fragmentos 
# más pequeños, genera embeddings utilizando un modelo de lenguaje y, finalmente, sube estos embeddings a Milvus, 
# una base de datos vectorial para su almacenamiento y consulta.

# Enlaces de interes: https://www.kaggle.com/datasets/ananyapam7/legalai/data
#
# Historial de revisiones
# 10/03/2025 - Creacion (primera version) del codigo
# 27/09/2025 - Implementación de librerías para vector database embedding

import os
from pick import pick
import sys
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from auxfunctions import load_objects,load_queries, visualize_docs, save_embeddings_cs, save_embeddings_q, all_underscores
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def cs_proc():            
  # --- 1. Load Data ---
  cases = load_objects("casedocs", "./archive/Object_casedocs")  
  statutes = load_objects("statutes", "./archive/Object_statutes")

  # --- 2. Prepare Documents for Chunking ---
  docs = []
  for case_id, case_text in cases.items():
      docs.append({"text": case_text, "type": "casedoc", "id": case_id})

  for statute_id, statute_text in statutes.items():
      docs.append({"text": statute_text, "type": "statute", "id": statute_id})

  # visualize_docs(docs) #Debug

  # --- 3. Chunk the Documents ---
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
  chunked_docs = []
  for doc in docs:
      chunks = text_splitter.split_text(doc["text"])
      for i, chunk in enumerate(chunks):
          chunked_docs.append({
              "text": chunk,
              "type": doc["type"],
              "id": doc["id"],
              "chunk": f"chunk{i}" 
          })

  #visualize_docs(chunks)
  print(f"Loaded {len(docs)} documents and split into {len(chunked_docs)} chunks.")

  # --- 4. Generate Embeddings ---

  #Let the user choose the embedding model to use
  title = 'Please choose your preferred embedding model: '
  options = ['all-mpnet-base-v2', 'all-MiniLM-L6-v2', 'multi-qa-mpnet-base-dot-v1', 'all-distilroberta-v1', 'back']
  

  selected_model = pick(options, title, indicator='=>', default_index=1)[0]

  if selected_model == 'back':
     return

  # Load a free embedding model (runs locally)
  print("Running " + selected_model + " embedding model...\n")
  model = SentenceTransformer(selected_model)

  def embed_text(texts):
      return model.encode(texts, 
      batch_size=64, 
      show_progress_bar=True, 
      convert_to_numpy=True)

  texts = [d["text"] for d in chunked_docs]
  embeddings = embed_text(texts)

  print("Saving the" + selected_model + "embeddings into .tsv files...\n")
  save_embeddings_cs(embeddings, chunked_docs, all_underscores(selected_model))


def q_proc():
  queries = load_queries("./archive/Query_doc.txt")
  test_queries = {
    "AILA_Q2": queries["AILA_Q2"],
    "AILA_Q5": queries["AILA_Q5"],
    "AILA_Q22": queries["AILA_Q22"],
  }
  # We are omitting the chunking since the queries are at around 1000 char long

  selected_models = ['all-mpnet-base-v2', 'all-MiniLM-L6-v2', 'multi-qa-mpnet-base-dot-v1', 'all-distilroberta-v1']
  def embed_text(texts):
    return model.encode(texts, 
    batch_size=64, 
    show_progress_bar=True, 
    convert_to_numpy=True)
  
  for model_name in selected_models:
    model = SentenceTransformer(f"sentence-transformers/{model_name}")
  
    # Prepare texts and IDs
    texts = list(test_queries.values())
    ids = list(test_queries.keys())
    embeddings = embed_text(texts)

    print("Saving the" + model_name + "embeddings into .tsv files...\n")
    save_embeddings_q(embeddings, texts,ids, all_underscores(model_name))


def main():
  while(True):
    title = 'Please choose what you wish to create the embeddings of: '
    options = ['Casedoc & Statutes', '3 Test queries for all models', 'Exit']

    selected_option = pick(options, title, indicator='=>', default_index=1)[0]
    if selected_option == options[2]:
      sys.exit()
    elif selected_option == options[0]:
      cs_proc()
    else:
      q_proc()
     

if __name__ == "__main__" :
  main()