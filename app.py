from flask import Flask, request, jsonify
from retriever import get_answer

app = Flask(__name__)

@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    user_input = data["question"]
    answer = get_answer(user_input)
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)
