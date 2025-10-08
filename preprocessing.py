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
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from auxfunctions import load_objects, visualize_docs, save_embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

            
# --- 1. Load Data ---
cases = load_objects("casedocs", "./archive/Object_casedocs")  
statutes = load_objects("statutes", "./archive/Object_statutes")

# queries = load_queries("./archive/Query_doc.txt")
# r_priorcases = load_judgments("./archive/relevance_judgments_priorcases.txt")
# r_statutes = load_judgments("./archive/relevance_judgments_statutes.txt")


# print("Case 1: " + str(cases["C1"]))
# print("Statute 200: " + str(statutes["S200"]))
# print("Query Q22: " + str(queries["AILA_Q22"]) + "\n")
# print("Relevance prriorcase Q4: " + str(r_priorcases["AILA_Q4"]) + "\n") 
# print("Relevance statute Q47" + str(r_statutes["AILA_Q47"])+ "\n") 

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
            "id": f"{doc['id']}",
            "chunk": f"chunk{i}"
        })

#visualize_docs(chunks)
print(f"Loaded {len(docs)} documents and split into {len(chunked_docs)} chunks.")

# --- 4. Generate Embeddings ---

#Let the user choose the embedding model to use


title = 'Please choose your preferred embedding model: '
options = ['all-mpnet-base-v2', 'all-MiniLM-L6-v2', 'multi-qa-mpnet-base-dot-v1', 'all-distilroberta-v1']

selected_model, index = pick(options, title, indicator='=>', default_index=1)


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

print("Saving the embeddings into .tsv files...\n")
save_embeddings(embeddings, chunked_docs, selected_model)
