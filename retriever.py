import os
import numpy as np
import openai
from dotenv import load_dotenv

from config import OPENAI_EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, OUTPUT_NPZ

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Load the compressed embeddings archive (built by embedding.py)
DB_NPZ_PATH = os.getenv("DB_NPZ_PATH", OUTPUT_NPZ)

_db = np.load(DB_NPZ_PATH, allow_pickle=True)
embeddings_matrix = _db["embeddings"].astype(np.float32)  # shape (N, D)
titles = _db["titles"]
plots = _db["plots"]
types = _db["types"] if "types" in _db.files else np.array(["movie"] * len(titles), dtype=object)
_norms = np.linalg.norm(embeddings_matrix, axis=1)

TYPE_LABELS = {"movie": "Movie", "game": "Video Game"}

def embed_text(text, model=OPENAI_EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS):
    response = openai.embeddings.create(
        input=text,
        model=model,
        dimensions=dimensions
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

MAX_HISTORY_TURNS = 12

def get_answer(question, top_k=5, history=None):
    # Embed the question (retrieval is always based on the latest question)
    question_embedding = embed_text(question)

    # Vectorized cosine similarity against every stored movie/game at once
    q_norm = np.linalg.norm(question_embedding)
    sims = embeddings_matrix @ question_embedding / (_norms * q_norm + 1e-10)

    top_idx = np.argpartition(-sims, top_k)[:top_k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    # Build context string from top matches
    context_blocks = [
        f"[{TYPE_LABELS.get(str(types[i]), 'Movie')}: {titles[i]}]: {plots[i]}" for i in top_idx
    ]
    context = "\n\n".join(context_blocks)

    system_prompt = f"""You are a helpful assistant that knows about my personal movie and video game collection.

The following context comes from my personal database of movies and video games that my family owns.
Each entry is tagged as either a Movie or a Video Game, and represents an item I have in one or more formats/platforms (DVD, Blu-Ray, 4K, Switch, PS5, etc), with information sourced from places like IMDb, Rotten Tomatoes, and RAWG. Use the conversation history to understand follow-up questions.

Context:
{context}"""

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        # history entries are {"role": "user"|"assistant", "content": "..."}
        messages.extend(history[-MAX_HISTORY_TURNS:])
    messages.append({"role": "user", "content": question})

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating answer: {str(e)}"
