import os
# 1. FIX: Disable memory limits BEFORE importing torch
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import glob
import torch

# --- CONFIG ---
BASE_DIR = "archive"
MODEL_NAME = "law-ai/InLegalBERT"
OUTPUT_PATH = "./models/InLegalBERT-AILA-Tuned"

# 2. FIX: Reduce Batch Size to fit in Mac Memory (Try 4, if it crashes, try 2)
BATCH_SIZE = 4 

def load_data():
    queries = {}
    with open(f"{BASE_DIR}/Query_doc.txt", 'r', errors='ignore') as f:
        for line in f:
            if "||" in line:
                qid, text = line.split("||")
                queries[qid.strip()] = text.strip()

    docs = {}
    # Only loading relevant docs to save RAM if needed, but loading all is usually fine for text
    for f in glob.glob(f"{BASE_DIR}/Object_casedocs/*.txt"):
        did = os.path.basename(f).replace(".txt", "")
        with open(f, 'r', errors='ignore') as file:
            docs[did] = file.read()

    train_examples = []
    with open(f"{BASE_DIR}/relevance_judgments_priorcases.txt", 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                qid, _, did, rel = parts
                if rel == '1' and qid in queries and did in docs:
                    # 3. OPTIMIZATION: Truncate text to 1000 chars to save VRAM
                    # LegalBERT usually only reads the first 512 tokens anyway.
                    train_examples.append(InputExample(texts=[queries[qid], docs[did][:2000]]))

    return train_examples

def main():
    # Check Device
    if torch.backends.mps.is_available():
        print("🚀 Using Mac GPU (MPS) with optimized memory settings.")
    else:
        print("⚠️ MPS not detected. Running on CPU (slow).")

    print("⚙️ Loading data...")
    train_examples = load_data()
    print(f"✅ Found {len(train_examples)} training pairs.")

    model = SentenceTransformer(MODEL_NAME)

    # DataLoader with smaller batch size
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    train_loss = losses.MultipleNegativesRankingLoss(model)

    print(f"🚀 Starting training (Batch Size: {BATCH_SIZE})...")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        output_path=OUTPUT_PATH,
        show_progress_bar=True
    )
    
    print(f"🎉 Model saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()