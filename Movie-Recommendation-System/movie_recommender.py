"""
Movie Recommendation System (Content-Based Filtering)
-------------------------------------------------------
Concepts used:
- Week 1: Python for AI development, data preprocessing
- Week 2: Machine Learning concepts (cosine similarity for recommendations)

This builds a content-based recommender: given a movie title, it suggests
similar movies based on genre overlap.

For a real dataset, download "movies.csv" from MovieLens:
https://grouplens.org/datasets/movielens/
and place it here with columns: title, genres
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# 1. Load Data
# ----------------------------
USE_CSV = False  # set True if you have movies.csv

if USE_CSV:
    df = pd.read_csv("movies.csv")
else:
    # Small built-in sample dataset for demo purposes
    data = {
        "title": [
            "The Dark Knight", "Inception", "Toy Story", "Finding Nemo",
            "The Notebook", "Titanic", "Avengers: Endgame", "Iron Man",
            "Frozen", "Up", "The Conjuring", "Get Out",
            "La La Land", "A Star is Born", "Interstellar", "The Martian"
        ],
        "genres": [
            "Action Crime Drama Thriller",
            "Action Sci-Fi Thriller Mystery",
            "Animation Comedy Family Adventure",
            "Animation Comedy Family Adventure",
            "Romance Drama",
            "Romance Drama",
            "Action Adventure Sci-Fi",
            "Action Adventure Sci-Fi",
            "Animation Family Musical Adventure",
            "Animation Adventure Comedy Family",
            "Horror Thriller Mystery",
            "Horror Thriller Mystery",
            "Romance Musical Drama",
            "Romance Musical Drama",
            "Sci-Fi Adventure Drama",
            "Sci-Fi Adventure Drama"
        ]
    }
    df = pd.DataFrame(data)

print(f"Dataset size: {len(df)} movies")

# ----------------------------
# 2. Data Preprocessing (Week 1)
# ----------------------------
df["genres"] = df["genres"].str.lower().str.replace("-", "")

# ----------------------------
# 3. Feature Extraction - convert genres into vectors
# ----------------------------
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(df["genres"])

# ----------------------------
# 4. Compute Similarity (Week 2 - ML concept: cosine similarity)
# ----------------------------
similarity_matrix = cosine_similarity(genre_matrix)

# ----------------------------
# 5. Recommendation Function
# ----------------------------
def recommend(movie_title, num_recommendations=3):
    if movie_title not in df["title"].values:
        return f"'{movie_title}' not found in dataset."

    idx = df[df["title"] == movie_title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # skip the first one (itself)
    recommendations = []
    for i, score in scores[1:num_recommendations + 1]:
        recommendations.append((df.iloc[i]["title"], round(score, 2)))

    return recommendations

# ----------------------------
# 6. Demo
# ----------------------------
print("\n--- Recommendations ---")
for movie in ["The Dark Knight", "Frozen", "Titanic", "Interstellar"]:
    recs = recommend(movie)
    print(f"\nBecause you liked '{movie}':")
    for title, score in recs:
        print(f"  -> {title} (similarity: {score})")
