from pymilvus import MilvusClient, utility, connections

#Renaming
# client = MilvusClient(
#     uri="http://localhost:19530",
#     token="root:Milvus"
# )

# client.rename_collection(
#     old_name="legal_docs",
#     new_name="legal_docs_all_MiniLM_L6_v2"
# )


#Deleting
connections.connect("default", host="localhost", port="19530")
print("Succesfully connected to milvus container!")

collection_name = "test_queries"

# 3. Drop (delete) the collection
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"✅ Collection '{collection_name}' deleted successfully.")
else:
    print(f"⚠️ Collection '{collection_name}' does not exist.")