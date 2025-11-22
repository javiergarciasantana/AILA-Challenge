import spacy
import json
import os

nlp = spacy.load('en_core_web_lg')

statutes_dir = '/Users/javiersantana/INF_2024_2025/Trabajo Fin de Grado/AILA-Challenge/archive/Object_statutes'
statutes = []

# Load all statutes from the directory
for filename in os.listdir(statutes_dir):
  if filename.endswith('.txt'):
    file_path = os.path.join(statutes_dir, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
      statutes.append({"id": filename, "content": file.read()})

statutes_vectors_list = []

# Process each statute
for statute in statutes:
  statute_id = statute["id"]
  content = statute["content"]

  # Split content into chunks
  doc = nlp(content)
  for i, sent in enumerate(doc.sents):
    chunk = sent.text
    vector = nlp(chunk).vector[:300].tolist()  # Reduce vector size

    entry = {"vector": vector, "textfile_id": statute_id, "chunk": chunk[:1000]}
    statutes_vectors_list.append(entry)

# Save the data to a JSON file
with open('statute_data.json', 'w', encoding='utf-8') as json_file:
  json.dump(statutes_vectors_list, json_file, indent=2, ensure_ascii=False)

print("JSON file created: statute_data.json")