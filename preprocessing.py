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
from auxfunctions import load_objects,load_queries, visualize_docs, save_embeddings_cs, save_embeddings_q, all_underscores, chunk_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from menu import Menu

def cs_proc(option, path):            
  # --- 1. Load Data ---
  input = load_objects(option, path)

  #Let the user choose the embedding model to use
  title = f'Please choose your preferred {option} embedding model: '
  options = ['BAAI/bge-base-en-v1.5', 'BAAI/bge-small-en', 'back']
  
  selected_model, model_num = pick(options, title, indicator='=>', default_index=1)

  if selected_model == 'back': return

  # --- 2. Prepare Documents for Chunking ---
  docs = []
  for case_id, case_text in input.items():
      docs.append({"text": case_text, "type": option, "id": case_id})

  # visualize_docs(docs) #Debug
  
  # all-mpnet-base-v2: 57972 chunks Casedocs | 514 chunks statutes
  # all-MiniLM-L6-v2: 88328 chunks Casedocs | 738 chunks statutes
  # multi-qa-mpnet-base-dot-v1: 38364 chunks Casedocs | 387 chunks statutes
  # all-distilroberta_v1: 38364 chunks Casedocs | 387 chunks statutes
  # bge-base-en-v1.5: 57634 chunks Casedocs | 505 chunks statutes

  # --- 3. Chunk the Documents (Optional) ---
  chunking_title = f'Do you want to chunk the {option} documents before embedding?'
  chunking_options = ['Yes', 'No']
  chunking_choice, _ = pick(chunking_options, chunking_title, indicator='=>', default_index=0)

  if chunking_choice == 'Yes':
      # chunk_sizes = [1500, 1000, 2200, 2200, 1000, 800]
      # chunk_overlaps = [250, 200, 300, 300, 200, 200]
      # print(chunk_sizes[model_num])
      # text_splitter = RecursiveCharacterTextSplitter(
      #     chunk_size=chunk_sizes[model_num],
      #     chunk_overlap=chunk_overlaps[model_num],  # Add overlap to maintain context between chunks
      #     length_function=len,
      # )
      chunked_docs = []
      for doc in docs:
          chunks = chunk_text(doc["text"])
          for i, chunk in enumerate(chunks):
              chunked_docs.append({
                  "text": chunk,
                  "type": doc["type"],
                  "id": doc["id"],
                  "chunk": f"chunk{i}" 
              })
      print(f"Loaded {len(docs)} documents and split into {len(chunked_docs)} chunks.")
  else:
      # No chunking, treat each document as a single chunk
      chunked_docs = []
      for doc in docs:
          chunked_docs.append({
              "text": doc["text"],
              "type": doc["type"],
              "id": doc["id"],
              "chunk": "chunk0"
          })
      print(f"Loaded {len(docs)} documents with no chunking.")

  # --- 4. Generate Embeddings ---
  print(f"Running {selected_model} embedding model for {option}...\n")
  model = SentenceTransformer(selected_model)

  def embed_text(texts):
      return model.encode(
          texts,
          show_progress_bar=True,
          normalize_embeddings=True if selected_model.startswith('BAAI') else False
  )

  # Update the "text" field in chunked_docs if using BAAI models to include the instruction
  if selected_model.startswith('BAAI'):
      if option == "casedoc":
          instruction = "Represent this case for retrieval: "
      elif option == "statute":
          instruction = "Represent this statute for retrieval: "
      for d in chunked_docs:
          d["text"] = instruction + d["text"]
      texts = [d["text"] for d in chunked_docs]
  else:
      texts = [d["text"] for d in chunked_docs]

  
  embeddings = embed_text(texts)
  print("Saving the" + selected_model + "embeddings into .tsv files...\n")
  save_embeddings_cs(embeddings, chunked_docs, option, all_underscores(selected_model))


def q_proc():
  queries = load_queries("./archive/Query_doc.txt")
  test_queries = {
    "AILA_Q2": queries["AILA_Q2"],
    "AILA_Q5": queries["AILA_Q5"],
    "AILA_Q22": queries["AILA_Q22"],
  }
  # We are omitting the chunking since the queries are at around 1000 char long

  selected_models = ['BAAI/bge-base-en-v1.5', 'BAAI/bge-small-en']
  def embed_text(texts):
    return model.encode(texts, 
    batch_size=64, 
    show_progress_bar=True, 
    convert_to_numpy=True,
    normalize_embeddings=True if model_name.startswith('BAAI') else False
  )
  
  for model_name in selected_models:
    model = SentenceTransformer(model_name)
  
    # Prepare texts and IDs
    texts = list(test_queries.values())

    if model_name.startswith('BAAI'):
      instruction = f"Represent this sentence for searching relevant passages: "
      texts = [instruction + t for t in texts]

    ids = list(test_queries.keys())
    embeddings = embed_text(texts)

    print("Saving the" + model_name + "embeddings into .tsv files...\n")
    save_embeddings_q(embeddings, texts,ids, all_underscores(model_name))


def preprocessing_menu():
    """Displays a menu to let the user choose which data to preprocess."""
    title = 'Please choose which data to preprocess:'
    options = [
        ('Case Documents', lambda: cs_proc("casedoc", "./archive/Object_casedocs")),
        ('Statutes', lambda: cs_proc("statute", "./archive/Object_statutes")),
        ('3 Test Queries for all models', q_proc),
        ('Back', None)
    ]
    
    # Use the Menu class to show the options
    menu = Menu(title, options)
    menu.show()
