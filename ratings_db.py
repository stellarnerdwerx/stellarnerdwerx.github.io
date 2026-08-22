"""Storage for logins and per-person movie ratings (seen / want to watch / liked).

Kept in its own sqlite file, separate from assets/stellarsearch.db, so that
redeploying the movie catalog (which gets overwritten by git pushes) never
touches the live ratings data sitting on the server.
"""

import os
import secrets

import sqlite3

from werkzeug.security import check_password_hash, generate_password_hash

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
# Anchored to this file's own folder (not the process cwd) so a console run of
# manage_users.py and the live WSGI process always agree on which file they mean.
RATINGS_DB_PATH = os.getenv("RATINGS_DB_PATH", os.path.join(_MODULE_DIR, "ratings.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    username TEXT NOT NULL REFERENCES users(username),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS movie_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL REFERENCES users(username),
    movie_id INTEGER NOT NULL,
    seen INTEGER NOT NULL DEFAULT 0,
    want_to_watch INTEGER NOT NULL DEFAULT 0,
    liked INTEGER NOT NULL DEFAULT 0,
    rated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(username, movie_id)
);
"""


def get_conn():
    conn = sqlite3.connect(RATINGS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def create_user(username, display_name, password):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO users (username, display_name, password_hash)
        VALUES (?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            display_name=excluded.display_name,
            password_hash=excluded.password_hash
        """,
        (username, display_name, generate_password_hash(password)),
    )
    conn.commit()
    conn.close()


def verify_login(username, password):
    conn = get_conn()
    # Case-insensitive: mobile keyboards auto-capitalize the first letter of text fields.
    row = conn.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", (username,)).fetchone()
    conn.close()
    if row and check_password_hash(row["password_hash"], password):
        return dict(row)
    return None


def create_session(username):
    token = secrets.token_urlsafe(32)
    conn = get_conn()
    conn.execute("INSERT INTO sessions (token, username) VALUES (?, ?)", (token, username))
    conn.commit()
    conn.close()
    return token


def get_user_for_token(token):
    if not token:
        return None
    conn = get_conn()
    row = conn.execute(
        """
        SELECT u.username, u.display_name
        FROM sessions s JOIN users u ON u.username = s.username
        WHERE s.token = ?
        """,
        (token,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_session(token):
    conn = get_conn()
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def list_users():
    conn = get_conn()
    rows = conn.execute("SELECT username, display_name FROM users ORDER BY display_name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def upsert_rating(username, movie_id, seen, want_to_watch, liked):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO movie_ratings (username, movie_id, seen, want_to_watch, liked)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(username, movie_id) DO UPDATE SET
            seen=excluded.seen,
            want_to_watch=excluded.want_to_watch,
            liked=excluded.liked,
            rated_at=CURRENT_TIMESTAMP
        """,
        (username, movie_id, int(bool(seen)), int(bool(want_to_watch)), int(bool(liked))),
    )
    conn.commit()
    conn.close()


def get_all_ratings():
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT r.movie_id, r.username, u.display_name, r.seen, r.want_to_watch, r.liked
        FROM movie_ratings r JOIN users u ON u.username = r.username
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_liked_counts():
    """Public-safe aggregate: how many people liked each movie, no usernames attached."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT movie_id, COUNT(*) AS liked_count FROM movie_ratings WHERE liked = 1 GROUP BY movie_id"
    ).fetchall()
    conn.close()
    return {str(r["movie_id"]): r["liked_count"] for r in rows}
