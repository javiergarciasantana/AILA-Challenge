import os, time
from pathlib import Path
import numpy as np
import pandas as pdz
from pymilvus import Collection, DataType, utility

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
    dir_path = Path("export")/model/kind
    dir_path.mkdir(parents=True, exist_ok=True)
    try:
        np.savetxt(dir_path/"embeddings.tsv", np.array(embeddings), delimiter="\t")
        md = pd.DataFrame({
            "id":[c["id"] for c in chunks],
            "chunk":[c["chunk"] for c in chunks],
            "type":[c["type"] for c in chunks],
            "text":[c["text"][:200].replace("\n"," ") for c in chunks]
        })
        md.to_csv(dir_path/"metadata.tsv", sep="\t", index=False)
        print("✅ Embeddings saved successfully.")
        print("✅ Metadata saved successfully.")
        return True
    except Exception as e:
        print(f"❌ Error while saving embeddings or metadata: {e}")
        return False

def save_embeddings_q(embeddings, texts, ids, model):
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

def expected_outcome(input_type, query_id, doc_id):
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
    input_file = "../archive/relevance_judgments_statutes.txt"
  elif input_type == "casedoc":
    input_file = "../archive/relevance_judgments_priorcases.txt"
  else:
    raise ValueError("Invalid input type. Use 'statute' or 'priorcase'.")

  # Open the file and process it
  with open(input_file, "r") as file:
    for line in file:
      # Match lines based on the query ID and input type
      pattern = rf"^{query_id} Q0 {doc_id} 1$"
      if re.match(pattern, line.strip()):
        print(line.strip())