# chatbot.py

import os
import openai
from sentence_transformers import SentenceTransformer
import chromadb

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- Config ---
CHROMA_DB_PATH = "/Users/lena/Documents/DATABASES/chroma_db"
COLLECTION_NAME = "faq_collection"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
OPENAI_MODEL = "gpt-3.5-turbo"  

# Get key from env
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
    
The following context comes from my personal database of movies that my family own. 
Each entry represents a movie I have in one or more formats (DVD, Blu-Ray, 4K, etc).

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
    temperature=0.3
)

    answer = response.choices[0].message.content
    return answer

def chat():
    model = load_model()
    collection = load_collection()

    print("🎬 MovieBot is ready. Ask anything about your movie collection!\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        context_chunks, _ = retrieve_context(user_input, model, collection)
        prompt = build_prompt(user_input, context_chunks)
        answer = ask_llm(prompt)
        print("\n🎥 MovieBot:", answer)
        print("-" * 60)

if __name__ == "__main__":
    chat()
