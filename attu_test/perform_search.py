import spacy
from pymilvus import connections, Collection, MilvusException

# Load the spaCy model for English language
spacy_model = spacy.load('en_core_web_lg')

# Connect to Milvus server
connections.connect(host="localhost", port="19530")

try:
      with open('/Users/javiersantana/INF_2024_2025/Trabajo Fin de Grado/AILA-Challenge/attu_test/Q2.txt', 'r') as file:
        # Read the content of Q2.txt
        user_input = file.read().strip()

        # Process user input using spaCy model to get embedding vector
        user_input_doc = spacy_model(user_input)
        user_vector = user_input_doc.vector[:300].tolist()

        #print(user_vector)

        # Define search parameters for similarity search
        search_params = {
            "metric_type": "L2",
            "offset": 0,
            "ignore_growing": False,
            "params": {"nprobe": 10}
        }

        # Connect to the Milvus collection named "Movies"
        collection = Collection("statute_spacy")

        # Perform similarity search using Milvus
        similarity_search_result = collection.search(
            data=[user_vector],
            anns_field="vector",
            param=search_params,
            limit=50,
            output_fields=['textfile_id']
        )

        # Display search results to the user
        for idx, hit in enumerate(similarity_search_result[0]):
            score = hit.distance
            textfile_id = hit.entity.textfile_id
            print(f"{idx + 1}. {textfile_id} (distance: {score})")

except MilvusException as e:
    # Handle Milvus exceptions
    print(e)
finally:
    # Disconnect from Milvus server
    connections.disconnect(alias="localhost")
    