import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import uuid #ids for chunks
from io import StringIO

# config
from config import PATH, DATA_PATH, CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

def load_data(file_path):
    """read in the movie data, ignore junk chars"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    df = pd.read_csv(StringIO(content))
    return df

def movie_to_text(row):
    """create text chunk per movie by combining relevant fields into string i.e. paragraph summarizing each movie """
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

def prepare_chunks(df):
    """prepare chunks from dataframe"""
    chunks = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        text_chunk = movie_to_text(row)
        chunks.append(text_chunk)


        metadatas.append({
            "title": row['title'],
            "year": row['year'],
            "imdb": row.get('imdb', None),
            "genres": row.get('genres', None),
            "director": row.get('director', None),
            # Include format info:
            "dvd": row.get('dvd', None),
            "blu_ray": row.get('blu-ray', None),
            "3d": row.get('3d', None),
            "burnt": row.get('burnt', None),
            "extra": row.get('extra', None),
            "digital": row.get('digital', None),
            "4k": row.get('4k', None),
            "runtime": row.get('runtime', None),
            "rating": row.get('rating', None),
        })

        ids.append(str(uuid.uuid4()))

    print(f"Prepared {len(chunks)} movie chunks.")
    return chunks, metadatas, ids

def generate_embeddings(chunks, model_name):
    """Generates embeddings for a list of text chunks."""
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Generating embeddings... This may take a moment.")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print(f"Generated {len(embeddings)} embeddings.")
    return embeddings

def store_in_chromadb(chunks, embeddings, metadatas, ids, db_path, collection_name):
    """Stores chunks, embeddings, and metadatas in ChromaDB."""
    print(f"Initializing ChromaDB client at {db_path}...")
    client = chromadb.PersistentClient(path=db_path)
    
    print(f"Getting or creating collection: {collection_name}...")
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"} # Using cosine similarity
    )
    
    print(f"Adding {len(chunks)} items to the collection...")
    # ChromaDB expects documents, embeddings, metadatas, and ids
    # Ensure embeddings are lists of floats, not numpy arrays 
    embeddings_list = [emb.tolist() for emb in embeddings]

    collection.add(
        documents=chunks,
        embeddings=embeddings_list,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Successfully added {collection.count()} items to '{collection_name}'.")

if __name__ == "__main__":
    print("Starting data ingestion process...")
    print("Following the pipeline: Spreadsheet → Chunks → Embeddings → Store")
    
    # 1. Load data
    df = load_data(DATA_PATH)
    
    if df is not None:
        # 2. Prepare chunks
        chunks, metadatas, ids = prepare_chunks(df)
        
        if chunks:
            # 3. Generate embeddings
            embeddings = generate_embeddings(chunks, EMBEDDING_MODEL)
            
            # 4. Store in ChromaDB
            store_in_chromadb(chunks, embeddings, metadatas, ids, CHROMA_DB_PATH, COLLECTION_NAME)
            
            print("Data ingestion process completed successfully!")
            print("Your knowledge base is ready for retrieval!")
        else:
            print("No chunks were prepared. Check your CSV format.")
    else:
        print("Failed to load data. Check your file path and format.")



import os
import json
import openai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === CONFIG ===
CSV_PATH = "media_inventory.csv"         # Your spreadsheet
OUTPUT_JSON = "media_embeddings.json"
MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)

# Choose columns to use as text context for embedding
TEXT_COLUMNS = ["title", "format", "genre", "platform", "year", "notes"]

# Create a full text blob per row
def combine_row(row):
    fields = [f"{col}: {str(row[col])}" for col in TEXT_COLUMNS if pd.notnull(row[col])]
    return " | ".join(fields)

df["combined_text"] = df.apply(combine_row, axis=1)
texts = df["combined_text"].tolist()

# === GENERATE EMBEDDINGS ===
all_embeddings = []

print(f"Generating embeddings for {len(texts)} items...")

for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = texts[i:i + BATCH_SIZE]
    try:
        response = openai.Embedding.create(
            model=MODEL,
            input=batch_texts
        )
        for j, result in enumerate(response["data"]):
            index = i + j
            all_embeddings.append({
                "original_row": df.iloc[index].to_dict(),  # Save all row info
                "text": texts[index],
                "embedding": result["embedding"]
            })
    except Exception as e:
        print(f"⚠️ Error processing batch {i}-{i+BATCH_SIZE}: {e}")

# === SAVE OUTPUT ===
with open(OUTPUT_JSON, "w") as f:
    json.dump(all_embeddings, f)

print(f"✅ Done! Saved {len(all_embeddings)} media embeddings to {OUTPUT_JSON}")

