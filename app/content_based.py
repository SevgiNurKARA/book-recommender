import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class ContentRecommender:
    def __init__(self, books_csv_path):
        self.books = pd.read_csv(books_csv_path)

        # Eğer 'description' sütunu yoksa title + authors'ı birleştirerek oluştur
        if 'description' not in self.books.columns:
            self.books['description'] = self.books['title'].fillna('') + " " + self.books['authors'].fillna('')

        # Boş açıklamaları doldur
        self.books['description'] = self.books['description'].fillna('')

        # TF-IDF vektörleştirme
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.books['description'])

        # Cosine benzerliği
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def get_recommendations(self, book_title, top_n=5):
        indices = pd.Series(self.books.index, index=self.books['title']).drop_duplicates()

        if book_title not in indices:
            return ["Book not found."]

        idx = indices[book_title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]  # Kendisi hariç

        book_indices = [i[0] for i in sim_scores]
        return self.books.iloc[book_indices][['title', 'authors']]

    def save_model(self, model_path):
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self, model_path)

# Test (örnek kullanım)
if __name__ == "__main__":
   model = ContentRecommender("data/books.csv")
   model.save_model("models/content_model.pkl")
   print("Model saved to models/content_model.pkl")