# main.py (Flask API)

from flask import Flask, request, jsonify
from hybrid import HybridRecommender
import pandas as pd

app = Flask(__name__)

# Load models and data
books_path = "data/books.csv"
ratings_path = "data/ratings.csv"
hybrid = HybridRecommender(books_path, ratings_path)

@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user_id")
    book_title = request.args.get("book_title")
    top_n = int(request.args.get("top_n", 5))

    if user_id is not None:
        try:
            user_id = int(user_id)
        except ValueError:
            return jsonify({"error": "User ID must be an integer."}), 400

    results = hybrid.recommend(book_title=book_title, user_id=user_id, top_n=top_n)
    if isinstance(results, list):  # Error or message
        return jsonify({"message": results})

    return jsonify(results.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
