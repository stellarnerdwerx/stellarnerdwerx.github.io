"""Sync movie_embeddings.npz with the movies and games_video tables in the db.

Diffs each source row's embed text against what's baked into the .npz and only
calls the OpenAI embeddings API for rows that are new or changed. Rows removed
from the db are dropped from the .npz. Untouched rows reuse their existing
embedding, so this stays cheap even as the catalog grows.

Usage:
    python reembed_changed.py            # sync and save
    python reembed_changed.py --dry-run  # just show what would change
"""

import sys
import sqlite3
import numpy as np

from config import OUTPUT_NPZ, DB_PATH, OPENAI_EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
from embedding import movie_to_text, game_to_text, get_embedding


def load_source_entries():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    movie_rows = conn.execute("SELECT * FROM movies").fetchall()
    game_rows = conn.execute("SELECT * FROM games_video").fetchall()
    conn.close()

    entries = {}
    for row in movie_rows:
        entries[("movie", row["id"])] = (row["title"], movie_to_text(row))
    for row in game_rows:
        entries[("game", row["id"])] = (row["title"], game_to_text(row))
    return entries


def load_existing_entries():
    data = np.load(OUTPUT_NPZ, allow_pickle=True)
    if "types" not in data.files or "source_ids" not in data.files:
        sys.exit(
            f"{OUTPUT_NPZ} predates type/source_id tracking (movie-only schema). "
            "Run embedding.py for a full rebuild first, then this script can sync incrementally."
        )

    titles, plots, embeddings = data["titles"], data["plots"], data["embeddings"]
    types, source_ids = data["types"], data["source_ids"]

    existing = {}
    for i in range(len(titles)):
        key = (str(types[i]), source_ids[i])
        existing[key] = {"title": str(titles[i]), "plot": str(plots[i]), "embedding": embeddings[i]}
    return existing


def main():
    dry_run = "--dry-run" in sys.argv

    source = load_source_entries()
    existing = load_existing_entries()

    new_keys = [k for k in source if k not in existing]
    changed_keys = [k for k in source if k in existing and existing[k]["plot"] != source[k][1]]
    removed_keys = [k for k in existing if k not in source]

    if not new_keys and not changed_keys and not removed_keys:
        print("Nothing changed — embeddings are already up to date.")
        return

    if new_keys:
        print(f"{len(new_keys)} new:")
        for kind, sid in new_keys:
            print(f"  - [{kind}] {source[(kind, sid)][0]}")
    if changed_keys:
        print(f"{len(changed_keys)} changed:")
        for kind, sid in changed_keys:
            print(f"  - [{kind}] {source[(kind, sid)][0]}")
    if removed_keys:
        print(f"{len(removed_keys)} removed (no longer in db):")
        for kind, sid in removed_keys:
            print(f"  - [{kind}] {existing[(kind, sid)]['title']}")

    if dry_run:
        print("\nDry run — no API calls made, .npz not modified.")
        return

    print("\nSyncing...")
    to_embed = new_keys + changed_keys
    for i, key in enumerate(to_embed, start=1):
        kind, sid = key
        title, text_chunk = source[key]
        embedding = get_embedding(text_chunk, OPENAI_EMBEDDING_MODEL, EMBEDDING_DIMENSIONS)
        existing[key] = {"title": title, "plot": text_chunk, "embedding": embedding}
        print(f"  [{i}/{len(to_embed)}] embedded [{kind}] {title}")

    for key in removed_keys:
        del existing[key]

    titles, plots, embeddings, types, source_ids = [], [], [], [], []
    for (kind, sid), val in existing.items():
        titles.append(val["title"])
        plots.append(val["plot"])
        embeddings.append(val["embedding"])
        types.append(kind)
        source_ids.append(sid)

    np.savez_compressed(
        OUTPUT_NPZ,
        embeddings=np.array(embeddings, dtype=np.float32),
        titles=np.array(titles, dtype=object),
        plots=np.array(plots, dtype=object),
        types=np.array(types, dtype=object),
        source_ids=np.array(source_ids, dtype=object),
    )
    print(
        f"\nSaved {len(existing)} total embeddings to {OUTPUT_NPZ} "
        f"({len(new_keys)} new, {len(changed_keys)} changed, {len(removed_keys)} removed)."
    )


if __name__ == "__main__":
    main()
