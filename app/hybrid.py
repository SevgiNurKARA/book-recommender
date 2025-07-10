from content_based import ContentRecommender
import pandas as pd
import joblib

class HybridRecommender:
    def __init__(self, books_path, collab_model_path):
        self.books = pd.read_csv(books_path)
        self.content_model = ContentRecommender(books_path)
        self.collab_model = joblib.load(collab_model_path)  # <-- Modeli yükle

    def recommend(self, book_title=None, user_id=None, top_n=5):
        if user_id is not None:
            try:
                recs = self.collab_model.recommend_for_user(user_id, self.books, top_n)
                if not recs.empty:
                    return recs
            except Exception as e:
                print("Collaborative filtering failed:", e)  # Hata detayını logla
                pass

        if book_title is not None:
            return self.content_model.get_recommendations(book_title, top_n)

        return ["Please provide either a book title or user ID"]
if __name__ == "__main__":
    hybrid = HybridRecommender("data/books.csv", "models/collab_model.pkl")
    print(hybrid.recommend(user_id=123))
    print(hybrid.recommend(book_title="The Hobbit"))
