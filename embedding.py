import os
import pandas as pd
import json
import openai
from dotenv import load_dotenv
from io import StringIO


load_dotenv()  # load OPENAI_API_KEY from .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Config
from config import PATH, DATA_PATH, CHROMA_DB_PATH, COLLECTION_NAME, OPENAI_EMBEDDING_MODEL, OUTPUT_JSON

def movie_to_text(row):
    """Combine movie info fields into a single text chunk for embedding."""
    owned_formats = []
    for fmt in ['dvd', 'blu-ray', '3d', 'burnt', 'extra', 'digital', '4k']:
        if str(row.get(fmt, "")).lower() in ["yes", "1", "true", "x"]:
            owned_formats.append(fmt.upper())

    formats = ", ".join(owned_formats) if owned_formats else "Unknown format"

    return (
        f"Title: {row['title']}. "
        f"Year: {row['year']}. "
        f"Genres: {row['genres']}. "
        f"Rating: {row['rating']}. "
        f"Plot: {row['plot']} "
        f"Cast: {row['cast']} "
        f"Director: {row['director']}. "
        f"Owned formats: {formats}."
    )

def get_embedding(text, model):
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


def main():
    """read in the movie data, ignore junk chars"""
    print(f"Loading CSV data from {DATA_PATH}...")
    with open(DATA_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    df = pd.read_csv(StringIO(content))

    print(f"Loaded {len(df)} rows")

    data = []
    print("Generating embeddings...")
    for i, row in df.iterrows():
        text_chunk = movie_to_text(row)
        embedding = get_embedding(text_chunk, OPENAI_EMBEDDING_MODEL)

        item = {
            "title": row['title'],
            "plot": text_chunk,
            "embedding": embedding
        }
        data.append(item)
        if (i+1) % 10 == 0:
            print(f"Processed {i+1} rows")

    print(f"Writing output to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f)

    print("Done!")

if __name__ == "__main__":
    main()
