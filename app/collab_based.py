import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

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

# Test (örnek kullanım):
#collab = CollabRecommender("data/ratings.csv")
#print(collab.recommend_for_user(123, pd.read_csv("data/books.csv")))
