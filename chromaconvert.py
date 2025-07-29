import chromadb
import json

db_path = "/Users/lena/Documents/DATABASES/chroma_db"  # your existing db file path
collection_name = "movieinfo_collection"

client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(collection_name)

# fetch all items (assuming small enough)
results = collection.get(include=['documents', 'metadatas', 'embeddings'])

data = []
for i in range(len(results['documents'])):
    item = {
        "title": results['metadatas'][i].get('title', 'Unknown'),
        "plot": results['documents'][i],
        # convert ndarray to list here
        "embedding": results['embeddings'][i].tolist(),
    }
    data.append(item)

with open('exported_movie_embeddings.json', 'w') as f:
    json.dump(data, f, indent=2)
