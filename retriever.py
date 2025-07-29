import os
import json
import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Load JSON file with documents and embeddings
DB_JSON_PATH = os.getenv("DB_JSON_PATH", "exported_movie_embeddings.json")

with open(DB_JSON_PATH, 'r') as f:
    database = json.load(f)

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def embed_text(text, model="text-embedding-3-large"):
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def get_answer(question, top_k=5):
    # Embed the question
    question_embedding = embed_text(question)

    # Calculate similarity scores against all items
    similarities = []
    for item in database:
        sim = cosine_similarity(question_embedding, item['embedding'])
        similarities.append((sim, item))

    # Get top k matches
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_matches = similarities[:top_k]

    # Build context string from top matches
    context_blocks = []
    for _, item in top_matches:
        context_blocks.append(f"[{item['title']}]: {item['plot']}")

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant that knows about my personal movie collection.

The following context comes from my personal database of movies that my family owns.
Each entry represents a movie I have in one or more formats (DVD, Blu-Ray, 4K, etc) and includes information sourced from places like IMDb and Rotten Tomatoes.

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer only based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating answer: {str(e)}"
