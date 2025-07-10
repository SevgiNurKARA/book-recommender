from content_based import ContentRecommender
from collab_based import CollabRecommender
import pandas as pd

class HybridRecommender:
    def __init__(self, books_path, ratings_path):
        self.books = pd.read_csv(books_path)
        self.content_model = ContentRecommender(books_path)
        self.collab_model = CollabRecommender(ratings_path)

    def recommend(self, book_title=None, user_id=None, top_n=5):
        if user_id is not None:
            try:
                recs = self.collab_model.recommend_for_user(user_id, self.books, top_n)
                if not recs.empty:
                    return recs
            except:
                pass  # Fallback to content-based

        if book_title is not None:
            return self.content_model.get_recommendations(book_title, top_n)

        return ["Please provide either a book title or user ID"]

    def save_model(self, model_path):
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        import joblib
        joblib.dump(self, model_path)

# Test (example usage):
if __name__ == "__main__":
    hybrid = HybridRecommender("data/books.csv", "data/ratings.csv")
    hybrid.save_model("models/hybrid_model.pkl")
    print("Model saved to models/hybrid_model.pkl")
    print(hybrid.recommend(user_id=123))
    print(hybrid.recommend(book_title="The Hobbit"))
