from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer

# 1. Conectar a Milvus
milvus_cli = MilvusClient(uri="http://localhost:19530")

# 2. Cargar los 10 primeros estatutos (puedes adaptar esta parte a tu formato)
estatutos = []
for i in range(1, 11):
    with open(f"./archive/Object_statutes/S{i}.txt", "r", encoding="utf-8") as f:
        estatutos.append(f.read())

# 3. Obtener embeddings con bge-small-en
model = SentenceTransformer("BAAI/bge-small-en")
embeddings = model.encode(estatutos, normalize_embeddings=True)

collection_name = "estatutos_poc"
if milvus_cli.has_collection(collection_name):
    milvus_cli.drop_collection(collection_name)

# 4. Definir el esquema de la colección
schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="doc_name", datatype=DataType.VARCHAR, max_length=200)
schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=embeddings.shape[1])
index_params = milvus_cli.prepare_index_params()

# Index for Dense
index_params.add_index(
  field_name="dense_vector", 
  index_type="HNSW", 
  metric_type="COSINE", 
  params={"M": 16, "efConstruction": 200}
)
# 5. Crear la colección
milvus_cli.create_collection(collection_name, schema=schema, index_params=index_params)

# 6. Insertar los embeddings
entities= []
for i in range (1, 11):
  entities.append({
      "doc_name": f'S{i}.txt',           
      "dense_vector": embeddings[i - 1]
  })
milvus_cli.insert(collection_name, entities)
milvus_cli.flush(collection_name)

# 7. Buscar usando el embedding del primer estatuto
results = milvus_cli.search(
    collection_name=collection_name,
    data=[embeddings[0]], # S1.txt
    anns_field="dense_vector",
    search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=10,
    output_fields=["doc_name"]
)

# 8. Mostrar resultados
print("Resultados de la búsqueda (doc_name, score):")
for hit in results[0]:
    # En MilvusClient, los output_fields se guardan dentro de la clave 'entity'
    # y la puntuación de similitud se guarda bajo la clave 'distance'
    doc_name = hit.get('entity', {}).get('doc_name', 'Desconocido')
    score = hit.get('distance', 0.0)
    print(f"Documento: {doc_name}, Score: {score:.4f}")

# 9. Limpiar (opcional)
milvus_cli.drop_collection(collection_name)