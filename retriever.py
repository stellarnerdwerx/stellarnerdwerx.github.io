from sentence_transformers import SentenceTransformer
import chromadb
import os
import openai

from config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K, OPENAI_MODEL

from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI client (make sure OPENAI_API_KEY is in your environment)
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# load embedding model
def load_model(model_name):
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)

# load ChromaDB
def load_collection(db_path, collection_name):
    print(f"Loading ChromaDB collection from {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    return collection

# global load on import
model = load_model(EMBEDDING_MODEL)
collection = load_collection(CHROMA_DB_PATH, COLLECTION_NAME)

def get_answer(question):
    """Runs semantic search and uses OpenAI to answer based on context."""
    results = search_movies(question, model, collection)

    if not results["documents"][0]:
        return "I couldn't find anything related to that question."

    # Build context from top documents
    context_blocks = []
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        title = metadata.get("title", "Unknown Title")
        context_blocks.append(f"[{title}]: {doc}")

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant that knows about my personal movie collection.

The following context comes from my personal database of movies that my family owns.
Each entry represents a movie I have in one or more formats (DVD, Blu-Ray, 4K, etc) and includes information sourced from places like IMDb and Rotten Tomatoes.

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,  # use string model name from config
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers based on the given context only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating answer: {str(e)}"

# query
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

# display results
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

if __name__ == "__main__":
    while True:
        user_input = input("\nAsk a movie-related question (or type 'exit'): ")
        if user_input.lower() in ['exit', 'quit']:
            break
        answer = get_answer(user_input)
        print("\nAnswer:")
        print(answer)
