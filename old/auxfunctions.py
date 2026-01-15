import os, time
from pathlib import Path
import numpy as np
import pandas as pd
from pymilvus import Collection, DataType, utility, connections, MilvusException
import sys
import re
import spacy


def milvus_connect() -> bool:
  try:
    connections.connect("default", host="localhost", port="19530")
    utility.list_collections()
    print("✅ Successfully connected to Milvus container!")
    return True
  except MilvusException as e:
    print(f"❌ Failed to connect to Milvus: {e}")
    return False

def read_file(p): 
    with open(p, 'r', encoding='utf-8') as f: return f.read()

def process_query_to_dict(line):
    parts = line.strip().split("||", 1)
    return {parts[0]: parts[1]} if len(parts) == 2 else {}

def process_judgment_to_dict(line, out):
    parts = line.strip().split()
    if len(parts)==4 and parts[3]=="1":
        k = parts[0]; out.setdefault(k, []).append(tuple(parts[1:3]))

def load_objects(kind, folder):
    if kind=="casedoc": rng, pref = range(1,2915), "C"
    elif kind=="statute": rng, pref = range(1,201), "S"
    else: return {}
    folder = Path(folder)
    return {f"{pref}{i}": read_file(folder/f"{pref}{i}.txt")
            for i in rng if (folder/f"{pref}{i}.txt").exists()}

def load_queries(path):
    q={}
    with open(path,'r',encoding='utf-8') as f:
        for line in f: q.update(process_query_to_dict(line))
    return q

def load_judgments(path):
    j={}
    with open(path,'r',encoding='utf-8') as f:
        for line in f: process_judgment_to_dict(line,j)
    return j

def visualize_docs(d):
    df = pd.DataFrame(d)
    pd.set_option("display.max_colwidth",100)
    print(df.head(10))

def all_underscores(name): return name.replace("-","_")

def save_embeddings_cs(embeddings, chunks, kind, model):
    if model.startswith('BAAI/'):
      model = model[5:]
    dir_path = Path("export")/model/kind
    dir_path.mkdir(parents=True, exist_ok=True)
    try:
        np.savetxt(dir_path/"embeddings.tsv", np.array(embeddings), delimiter="\t")
        md = pd.DataFrame({
            "id":[c["id"] for c in chunks],
            "chunk":[c["chunk"] for c in chunks],
            "type":[c["type"] for c in chunks],
            "text":[c["text"].replace("\n"," ") for c in chunks]
        })
        md.to_csv(dir_path/"metadata.tsv", sep="\t", index=False)
        print("✅ Embeddings saved successfully.")
        print("✅ Metadata saved successfully.")
        return True
    except Exception as e:
        print(f"❌ Error while saving embeddings or metadata: {e}")
        return False

def save_embeddings_q(embeddings, texts, ids, model):
    if model.startswith("BAAI/"):
      model = model[5:]
    dir_path = Path("export/queries")/model
    dir_path.mkdir(parents=True, exist_ok=True)
    try:
        np.savetxt(dir_path/"embeddings.tsv", np.array(embeddings), delimiter="\t")
        pd.DataFrame({"id":ids,"model":model,"text":texts}).to_csv(dir_path/"metadata.tsv", sep="\t", index=False)
        print("✅ Embeddings saved successfully.")
        print("✅ Metadata saved successfully.")
        print(f"✅ Done with {model}!")
        return True
    except Exception as e:
        print(f"❌ Error while saving embeddings or metadata of queries: {e}")
        return False

def load_embeddings(model, kind=''):
    dir_path = Path("export")/model/kind
    try:
        meta = pd.read_csv(dir_path/"metadata.tsv", sep="\t")
        emb = np.loadtxt(dir_path/"embeddings.tsv", delimiter="\t")
        return meta, emb
    except Exception as e:
        print(f"❌ Error while loading embeddings or metadata: {e}")
        return False

def _metric_for_model(model_name):
    return "IP" if "dot" in (model_name or "").lower() else "COSINE"

def ensure_index(collection_name, model_name, index_type="HNSW"):
    col = Collection(collection_name)
    metric = _metric_for_model(model_name)
    it = index_type.upper()
    if not col.has_index():
        if it=="HNSW":
            col.create_index("embedding", {
                "index_type":"HNSW","metric_type":metric,
                "params":{"M":16,"efConstruction":200}
            })
        else:
            col.create_index("embedding", {
                "index_type":"FLAT","metric_type":metric,"params":{}
            })
    col.load()
    return {"metric_type":metric,"params":({"ef":64} if it=="HNSW" else {})}

def visualize_collections():
    cols = utility.list_collections()
    print("\n📦 Collections in Milvus:")
    for i,c in enumerate(cols,1): print(f"  {i}. {c}")
    print("\n✨ Total collections:", len(cols))
    time.sleep(5)

def average_chars_in_textfiles(dir_path):
    total = count = 0
    for fn in os.listdir(dir_path):
        p = os.path.join(dir_path, fn)
        if os.path.isfile(p) and fn.endswith(".txt"):
            with open(p,'r',encoding='utf-8') as f:
                total += len("".join(f.read().split()))
                count += 1
    if not count:
        print("No text files found in the directory."); return 0
    avg = total / count
    print(f"Char number avg in {dir_path}: {avg}")
    return avg

def is_expected(input_type, query_id, doc_id):
  """
  Process the input file based on the input type (statute or priorcase) 
  and query ID to match specific lines.

  Args:
    input_type (str): Either "statute" or "priorcase".
    query_id (str): The query ID or case/statute ID to match in the regex.

  Returns:
    None
  """
  # Determine the input file based on the input type
  if input_type == "statute":
    input_file = "./archive/relevance_judgments_statutes.txt"
  elif input_type == "casedoc":
    input_file = "./archive/relevance_judgments_priorcases.txt"
  else:
    raise ValueError("Invalid input type. Use 'statute' or 'priorcase'.")

  # Open the file and process it
  with open(input_file, "r") as file:
    for line in file:
      # Match lines based on the query ID and input type
      pattern = rf"^{query_id} Q0 {doc_id} 1$"
      if re.match(pattern, line.strip()):
        return True
  
  return False


def count_expected(input_type, query_id):
  counter = 0
  # Determine the input file based on the input type
  if input_type == "statute":
    input_file = "./archive/relevance_judgments_statutes.txt"
  elif input_type == "casedoc":
    input_file = "./archive/relevance_judgments_priorcases.txt"
  else:
    raise ValueError("Invalid input type. Use 'statute' or 'priorcase'.")

  # Open the file and process it
  with open(input_file, "r") as file:
    for line in file:
      # Match lines based on the query ID and input type
      pattern = rf"^{query_id} Q0 .+ 1$"
      if re.match(pattern, line.strip()):
        counter += 1
  
  return counter

# After collecting rows_c (list of dicts with "id" like "C179 chunk2")
def aggregate_by_case(rows):
    from collections import defaultdict
    agg = defaultdict(list)
    for row in rows:
        case_id = row["id"].split()[0]  # or use regex if needed
        agg[case_id].append(row)
    # Keep the best scoring chunk per case
    best_per_case = [max(chunks, key=lambda x: x["score"]) for chunks in agg.values()]
    # Sort by score descending
    return sorted(best_per_case, key=lambda x: x["score"], reverse=True)


def chunk_text(text, chunk_size=1000, overlap=200):
    import spacy
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = current[-overlap:] + " " + sent
    if current:
        chunks.append(current.strip())
    return chunks


from sentence_transformers import CrossEncoder

# 1. Global variable to store the model instance (Singleton Pattern)
_reranker_model = None

def get_reranker():
    """
    Loads the model only once.
    Using 'BAAI/bge-reranker-v2-m3' for better legal reasoning and longer context support.
    """
    global _reranker_model
    if _reranker_model is None:
        print("Loading Reranker Model (this happens only once)...")
        # 'BAAI/bge-reranker-v2-m3' is state-of-the-art and supports longer sequences.
        # We set max_length=1024 to capture more text than the standard 512 limit.
        try:
            _reranker_model = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=1024, trust_remote_code=True)
        except:
            # Fallback to 'large' if v2-m3 causes environment issues
            print("Falling back to bge-reranker-large...")
            _reranker_model = CrossEncoder('BAAI/bge-reranker-large', max_length=512)
    return _reranker_model

def rerank_with_cross_encoder(query, candidates, top_n=10):
    """
    Reranks candidates using a CrossEncoder.
    """
    if not candidates:
        return []

    # Load model (cached)
    cross_encoder = get_reranker()

    # Prepare pairs: (Query, Document Text)
    # Ensure text is a string to avoid errors
    pairs = [(str(query), str(cand.get('text', ''))) for cand in candidates]

    # Predict scores
    scores = cross_encoder.predict(pairs)

    # Assign scores back to candidates
    for cand, score in zip(candidates, scores):
        cand['rerank_score'] = score

    # Sort all candidates by rerank_score (descending)
    reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

    # Cleanup helper field
    for cand in reranked:
        cand.pop('rerank_score', None)

    return reranked[:top_n]










  #  def rerank_with_cross_encoder(query, candidates, top_n=10):
  #   # Store original indices
  #   for idx, cand in enumerate(candidates):
  #       cand['orig_rank'] = idx

  #   cross_encoder = CrossEncoder('BAAI/bge-reranker-base')
  #   pairs = [(query, cand['text']) for cand in candidates]
  #   scores = cross_encoder.predict(pairs)
  #   for cand, score in zip(candidates, scores):
  #       cand['rerank_score'] = score

  #   # Sort by rerank_score (descending)
  #   reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

  #   # Build final list: keep candidate at original position if its orig_rank < reranked position
  #   final = [None] * len(candidates)
  #   used = set()
  #   for new_idx, cand in enumerate(reranked):
  #       orig_idx = cand['orig_rank']
  #       flag = cand['id'][-1]
  #       if orig_idx <= new_idx and orig_idx not in used and flag == '✧':
  #           final[orig_idx] = cand
  #           used.add(orig_idx)
  #       else:
  #           # Find next available slot from new_idx onwards
  #           for i in range(new_idx, len(candidates)):
  #               if final[i] is None:
  #                   cand["id"] += "↑"
  #                   final[i] = cand
  #                   break

  #   # Remove helper field and trim to top_n
  #   final = [cand for cand in final if cand is not None]
  #   for cand in final:
  #       cand.pop('orig_rank', None)
  #       cand.pop('rerank_score', None)
  #   return final[:top_n]