import pandas as pd
import pandas as pd
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import train_test_split

class CollabRecommender:
    def __init__(self, ratings_csv_path):
        self.ratings_df = pd.read_csv(ratings_csv_path)
        reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(self.ratings_df[['user_id', 'book_id', 'rating']], reader)
        self.trainset, self.testset = train_test_split(self.data, test_size=0.2)
        self.model = SVD()
        self.model.fit(self.trainset)

    def evaluate(self):
        predictions = self.model.test(self.testset)
        return accuracy.rmse(predictions)

    def recommend_for_user(self, user_id, book_df, n=5):
        # Sadece daha önce oy verilmemiş kitapları değerlendir
        rated_books = self.ratings_df[self.ratings_df['user_id'] == user_id]['book_id'].tolist()
        all_books = book_df['book_id'].unique()
        unseen_books = [bid for bid in all_books if bid not in rated_books]

        predictions = [
            (book_id, self.model.predict(user_id, book_id).est)
            for book_id in unseen_books
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_books = predictions[:n]
        return book_df[book_df['book_id'].isin([bid for bid, _ in top_books])][['title', 'authors']]

    def save_model(self, model_path):
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        import joblib
        joblib.dump(self, model_path)

# Test (örnek kullanım):
if __name__ == "__main__":
    collab = CollabRecommender("data/ratings.csv")
    collab.save_model("models/collab_model.pkl")
    print("Model saved to models/collab_model.pkl")
