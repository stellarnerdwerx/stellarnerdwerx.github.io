import os
import sqlite3
import numpy as np
import openai
from dotenv import load_dotenv


load_dotenv()  # load OPENAI_API_KEY from .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Config
from config import OPENAI_EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, OUTPUT_NPZ, DB_PATH

FORMAT_COLUMNS = {
    "dvd": "DVD",
    "bluray": "BLU-RAY",
    "three_d": "3D",
    "burnt": "BURNT",
    "digital": "DIGITAL",
    "four_k": "4K",
}

GAME_PLATFORM_COLUMNS = {
    "lightguncon": "Light Gun Controller",
    "N64": "N64",
    "NES": "NES",
    "SNES": "SNES",
    "GAMECUBE": "GameCube",
    "SWITCH": "Switch",
    "WII": "Wii",
    "WII_U": "Wii U",
    "GAMEBOY": "Game Boy",
    "GAMEBOY_COLOR": "Game Boy Color",
    "GAMEBOY_ADVANCE": "Game Boy Advance",
    "DS": "DS",
    "THREE_DS": "3DS",
    "PS1": "PS1",
    "PS2": "PS2",
    "PS3": "PS3",
    "PS4": "PS4",
    "PS4_VR": "PS4 VR",
    "PS5": "PS5",
    "PSP": "PSP",
    "PS_VITA": "PS Vita",
    "SEGA_CD": "Sega CD",
    "SEGA_GENESIS": "Sega Genesis",
    "XBOX": "Xbox",
    "XBOX_360": "Xbox 360",
    "XBOX_ONE": "Xbox One",
    "XBOX_KINECT": "Xbox Kinect",
    "SEGA_GAMEGEAR": "Sega Game Gear",
    "TENGEN": "Tengen",
}

def movie_to_text(row):
    """Combine movie info fields into a single text chunk for embedding."""
    owned_formats = [label for col, label in FORMAT_COLUMNS.items() if str(row[col]) == "1"]
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

def game_to_text(row):
    """Combine video game info fields into a single text chunk for embedding."""
    owned_platforms = [label for col, label in GAME_PLATFORM_COLUMNS.items() if str(row[col]) == "1"]
    platforms = ", ".join(owned_platforms) if owned_platforms else "Unknown platform"

    return (
        f"Title: {row['title']}. "
        f"Genre: {row['genre'] or 'Unknown genre'}. "
        f"Rating: {row['rating'] or 'Unrated'}. "
        f"Plot: {row['plot'] or 'No description available.'} "
        f"Owned platforms: {platforms}."
    )

def get_embedding(text, model, dimensions):
    response = openai.embeddings.create(
        input=text,
        model=model,
        dimensions=dimensions
    )
    return response.data[0].embedding


def main():
    print(f"Loading movies and video games from {DB_PATH}...")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    movie_rows = conn.execute("SELECT * FROM movies").fetchall()
    game_rows = conn.execute("SELECT * FROM games_video").fetchall()
    conn.close()

    print(f"Loaded {len(movie_rows)} movies, {len(game_rows)} video games")

    entries = [("movie", row, movie_to_text) for row in movie_rows]
    entries += [("game", row, game_to_text) for row in game_rows]

    titles = []
    plots = []
    vectors = []
    types = []
    source_ids = []
    print("Generating embeddings...")
    for i, (kind, row, to_text) in enumerate(entries):
        text_chunk = to_text(row)
        embedding = get_embedding(text_chunk, OPENAI_EMBEDDING_MODEL, EMBEDDING_DIMENSIONS)

        titles.append(row["title"])
        plots.append(text_chunk)
        vectors.append(embedding)
        types.append(kind)
        source_ids.append(row["id"])
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(entries)}")

    print(f"Writing output to {OUTPUT_NPZ}...")
    np.savez_compressed(
        OUTPUT_NPZ,
        embeddings=np.array(vectors, dtype=np.float32),
        titles=np.array(titles, dtype=object),
        plots=np.array(plots, dtype=object),
        types=np.array(types, dtype=object),
        source_ids=np.array(source_ids, dtype=object),
    )

    print("Done!")

if __name__ == "__main__":
    main()
