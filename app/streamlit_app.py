import streamlit as st
import pandas as pd
from hybrid import HybridRecommender
import joblib
import os

st.set_page_config(page_title="📚 Book Recommendation System", layout="centered")
st.title("📚 Book Recommender")
st.write("Get book suggestions based on content or user preferences.")

# ⏱️ MODEL CACHE
@st.cache_resource
def load_hybrid_model():
    model_path = "models/hybrid_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        books_path = "data/books.csv"
        ratings_path = "data/ratings.csv"
        return HybridRecommender(books_path, ratings_path)

hybrid = load_hybrid_model()  # <--- cache’lenmiş versiyonu çağır

# Sidebar
st.sidebar.header("Input Options")
user_id = st.sidebar.text_input("User ID (for collaborative filtering)")
book_title = st.sidebar.text_input("Book Title (for content-based)")
top_n = st.sidebar.slider("How many books to recommend?", min_value=1, max_value=10, value=5)

# Button
if st.sidebar.button("Get Recommendations"):
    with st.spinner("Fetching recommendations..."):
        if user_id:
            try:
                user_id = int(user_id)
            except ValueError:
                st.error("User ID must be a number.")
                st.stop()
        results = hybrid.recommend(book_title=book_title, user_id=user_id, top_n=top_n)

        if isinstance(results, pd.DataFrame):
            st.success("Here are your recommendations:")
            st.dataframe(results)
        else:
            st.warning(results[0])
else:
    st.info("Enter a book title or user ID to begin.")
