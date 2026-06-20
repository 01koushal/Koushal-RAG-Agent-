from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_backend import ask_koushal

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Koushal AI Backend Running"

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    query = data["query"]
    answer = ask_koushal(query)

    return jsonify({
        "question": query,
        "answer": answer
    })

if __name__ == "__main__":
    app.run(debug=True)