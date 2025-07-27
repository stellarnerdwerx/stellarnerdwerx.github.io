# retriever.py

from sentence_transformers import SentenceTransformer
import chromadb
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration ---
CHROMA_DB_PATH = "/Users/lena/Documents/DATABASES/chroma_db"
COLLECTION_NAME = "faq_collection"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5  # Number of results to return

# --- Load embedding model ---
def load_model(model_name):
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)

# --- Load ChromaDB collection ---
def load_collection(db_path, collection_name):
    print(f"Loading ChromaDB collection from {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    return collection

# --- Query function ---
def search_movies(query, model, collection, top_k=TOP_K):
    print(f"Encoding query: {query}")
    query_embedding = model.encode([query])[0]  # single query

    print(f"Searching top {top_k} results...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    return results

# --- Display function ---
def display_results(results):
    print("\n--- Top Results ---\n")
    for i in range(len(results["documents"][0])):
        print(f"Result #{i+1}:")
        print(f"Movie summary: {results['documents'][0][i]}")
        metadata = results["metadatas"][0][i]
        print(f"  Title: {metadata.get('title')}")
        print(f"  Year: {metadata.get('year')}")
        print(f"  Genre: {metadata.get('genres')}")
        print(f"  Formats: DVD: {metadata.get('dvd')}, Blu-ray: {metadata.get('blu_ray')}, 4K: {metadata.get('4k')}")
        print(f"  Distance: {round(results['distances'][0][i], 4)}")
        print("-" * 40)

# --- Main block ---
if __name__ == "__main__":
    model = load_model(EMBEDDING_MODEL)
    collection = load_collection(CHROMA_DB_PATH, COLLECTION_NAME)

    while True:
        user_input = input("\nAsk a movie-related question (or type 'exit'): ")
        if user_input.lower() in ['exit', 'quit']:
            break
        results = search_movies(user_input, model, collection)
        display_results(results)
