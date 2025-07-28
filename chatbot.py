# chatbot.py

import os
import openai
from sentence_transformers import SentenceTransformer
import chromadb

# quiet annoying warnings
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# config
from config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K, OPENAI_MODEL

# Get key from env
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)

def retrieve_context(query, model, collection, top_k=TOP_K):
    query_embedding = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    return results["documents"][0], results["metadatas"][0]

def build_prompt(query, context_chunks):
    context_text = "\n\n".join(context_chunks)
    return f"""You are a helpful assistant that knows about my personal movie collection.
    
The following context comes from my personal database of movies that my family owns. 
Each entry represents a movie I have in one or more formats (DVD, Blu-Ray, 4K, etc) and includes information sourced from places like IMDb and Rotten Tomatoes.

Context:
{context_text}

Question: {query}

Answer:"""


def ask_llm(prompt, model=OPENAI_MODEL):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that answers based on the given context only."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7
)

    answer = response.choices[0].message.content
    return answer

def chat():
    model = load_model()
    collection = load_collection()

    print("🎬 The Curator is ready. Ask anything about your movie collection!\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break

        context_chunks, _ = retrieve_context(user_input, model, collection)
        prompt = build_prompt(user_input, context_chunks)
        answer = ask_llm(prompt)
        print("\n🎥 The Curator:", answer)
        print("-" * 60)

if __name__ == "__main__":
    chat()
