from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility
from auxfunctions import ensure_index
import numpy as np
import time

class CaseStatuteStorer:
    """Manages storage of casedoc and statute embeddings in Milvus."""

    def __init__(self, model_name, kind):
        """
        Initializes the storer for a specific embedding model.
        Args:
            model_name (str): The name of the model.
        """
        self.model_name = model_name
        self.kind = kind
        self.collection_name = f"{self.kind}_{self.model_name}"

    def store(self, meta_df, embeddings):
      """Creates a collection and inserts casedoc/statute data."""


      # Normalize embeddings for cosine similarity (IP)
      #embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

      if utility.has_collection(self.collection_name):
          print(f"⚠️ Collection '{self.collection_name}' already exists. Skipping data insertion.")
          return

      print(f"🧱 Creating collection '{self.collection_name}'...")
      _, dim = embeddings.shape
      print(f"Embedding dimension: {dim}")


      fields = [
          FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=200),
          FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=20),
          FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=20),
          FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
          FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=30000),
      ]
      schema = CollectionSchema(fields, description=f"{self.kind} embeddings")
      collection = Collection(self.collection_name, schema)

      print("🚀 Inserting data in batches...")
      batch_size = 500
      for i in range(0, len(embeddings), batch_size):
          batch_ids = meta_df["id"][i:i+batch_size].tolist()
          batch_chunks = meta_df["chunk"][i:i+batch_size].tolist()
          batch_types = meta_df["type"][i:i+batch_size].tolist()
          batch_embeddings = embeddings[i:i+batch_size].tolist()
          batch_texts = meta_df["text"][i:i+batch_size].tolist()

          collection.insert([batch_ids, batch_chunks, batch_types, batch_embeddings, batch_texts])

      collection.flush()
      print(f"✅ Inserted {collection.num_entities} entities into '{self.collection_name}'.")

      # Create optimized index
      # ensure_index(self.collection_name, self.model_name, index_type="IV_FLAT")
      print(f"🏁 Data inserted and indexed in '{self.collection_name}'.")



class QueryStorer:
    """Manages storage of query embeddings in Milvus."""

    def __init__(self, model_name):
        """
        Initializes the storer for a specific embedding model.
        Args:
            model_name (str): The name of the model.
        """
        self.model_name = model_name

    def store_test_queries(self, meta_df, embeddings):
        """
        Creates a collection and inserts test query data.
        This can be expanded to handle different query types by adding more methods.
        """
        collection_name = f"queries_{self.model_name}"
        if utility.has_collection(collection_name):
            print(f"Collection '{collection_name}' already exists. Skipping.")
            return

        print(f"Creating collection '{collection_name}'...")
        _, dim = embeddings.shape
        # Normalize embeddings for inner product metric
        #embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=20),
            FieldSchema(name="model", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=30000),
        ]
        schema = CollectionSchema(fields, description="Test queries embeddings")
        collection = Collection(collection_name, schema)

        ids = meta_df["id"].tolist()
        model_col = meta_df["model"].tolist()
        text_col = meta_df["text"].tolist()

        collection.insert([ids, model_col, embeddings.tolist(), text_col])
        collection.flush()
        ensure_index(collection_name, self.model_name, index_type="FLAT")

        print(f"Test queries inserted into '{collection_name}'.")
        time.sleep(1) # Brief pause