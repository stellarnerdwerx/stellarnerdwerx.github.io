from flask import Flask, request, jsonify
from flask_cors import CORS

from retriever import get_answer
import ratings_db
import os
from dotenv import load_dotenv

project_folder = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(project_folder, '.env'))

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
CORS(app)  # Enables CORS so your GitHub Pages frontend can call this API

ratings_db.init_db()


def _current_user():
    auth_header = request.headers.get("Authorization", "")
    token = auth_header[7:] if auth_header.startswith("Bearer ") else None
    return ratings_db.get_user_for_token(token)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    user_input = data["question"]
    history = data.get("history", [])
    answer = get_answer(user_input, history=history)
    return jsonify({"response": answer})


@app.route("/users", methods=["GET"])
def users():
    # Requires login so an anonymous visitor can't enumerate who has an account.
    if not _current_user():
        return jsonify({"error": "Not logged in"}), 401
    return jsonify(ratings_db.list_users())


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    user = ratings_db.verify_login(data.get("username", ""), data.get("password", ""))
    if not user:
        return jsonify({"error": "Invalid username or password"}), 401
    token = ratings_db.create_session(user["username"])
    return jsonify({
        "token": token,
        "username": user["username"],
        "display_name": user["display_name"],
    })


@app.route("/logout", methods=["POST"])
def logout():
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        ratings_db.delete_session(auth_header[7:])
    return jsonify({"ok": True})


@app.route("/ratings/liked-counts", methods=["GET"])
def ratings_liked_counts():
    # Public: just a count per movie (0/1/2), never who -- safe for anyone browsing the site.
    return jsonify(ratings_db.get_liked_counts())


@app.route("/ratings", methods=["GET"])
def get_ratings():
    if not _current_user():
        return jsonify({"error": "Not logged in"}), 401
    return jsonify(ratings_db.get_all_ratings())


@app.route("/ratings", methods=["POST"])
def set_rating():
    user = _current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json() or {}
    movie_id = data.get("movie_id")
    if not movie_id:
        return jsonify({"error": "movie_id is required"}), 400

    ratings_db.upsert_rating(
        user["username"],
        movie_id,
        seen=data.get("seen", False),
        want_to_watch=data.get("want_to_watch", False),
        liked=data.get("liked", False),
    )
    return jsonify({"ok": True})

# No app.run() needed for deployment on PythonAnywhere
