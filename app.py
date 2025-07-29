from flask import Flask, request, jsonify
from retriever import get_answer
import os
from dotenv import load_dotenv

project_folder = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(project_folder, '.env'))

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    user_input = data["question"]
    answer = get_answer(user_input)
    return jsonify({"response": answer})

# No app.run() needed for deployment on PythonAnywhere
