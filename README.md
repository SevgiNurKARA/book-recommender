# 📚 Book Recommender System

A hybrid book recommendation system using both content-based and collaborative filtering, built with Python, Streamlit, and Flask.

## 🚀 Features

* 📖 **Content-Based Filtering**: Recommends similar books based on descriptions and metadata using TF-IDF and cosine similarity.
* 🤝 **Collaborative Filtering**: Suggests books using user ratings and the Surprise SVD model.
* ⚡ **Hybrid Mode**: Combines both methods to improve recommendation quality.
* 🎛 **Interactive UI**: Built with Streamlit to provide a simple and clean user interface.
* 🌐 **API Access**: Flask-based API for integration and testing.

---

## 📂 Project Structure

```
book-recommender/
├── app/
│   ├── content_based.py         # TF-IDF based recommender
│   ├── collab_based.py          # Collaborative filtering with Surprise
│   ├── hybrid.py                # Hybrid logic
│   ├── streamlit_app.py         # Streamlit frontend
│   └── main.py                  # Flask API
├── data/
│   ├── books.csv                # Book metadata
│   └── ratings.csv              # User ratings
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## ⚙️ Installation

```bash
# Clone the repository
https://github.com/SevgiNurKara/book-recommender.git
cd book-recommender

# Install dependencies
pip install -r requirements.txt
```

---

## 🎛 Run Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Use the sidebar to enter a book title or user ID and get top N recommendations.

---

## 🌐 Run Flask API

```bash
python app/main.py
```

Then open your browser:

```
http://localhost:5000/recommend?book_title=The%20Hobbit
http://localhost:5000/recommend?user_id=123
```

---

## 🧪 Example Datasets

This project uses the [Goodbooks-10k dataset](https://github.com/zygmuntz/goodbooks-10k) for books and ratings.
Make sure to place `books.csv` and `ratings.csv` under the `/data` folder.

---

## 🧠 Future Improvements

* 🔍 Search by genre or author
* 🧾 Explanation interface (SHAP/LIME)
* ☁️ Cloud deployment (Streamlit Cloud / Render)

---

## 👤 Author

**Sevgi Nur Kara**
AI & Data Science Enthusiast
[GitHub](https://github.com/SevgiNurKARA) | [LinkedIn](https://linkedin.com/in/sevginurkara)

---

## 📜 License

MIT License
