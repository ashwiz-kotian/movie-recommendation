import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 📁 Load Dataset
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
movies_path = os.path.join(BASE_DIR, 'data/movies.csv')

movies = pd.read_csv(movies_path)

# Keep required columns
movies = movies[['movieId', 'title', 'genres']]

# Remove duplicates
movies = movies.drop_duplicates(subset='title').reset_index(drop=True)

# Clean genres
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

# Create tags
movies['tags'] = movies['genres']

# -------------------------------
# 🧠 Vectorization
# -------------------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)

# -------------------------------
# 🎬 Recommendation Function
# -------------------------------
def recommend(movie):
    movie = movie.lower()

    # Flexible search (partial + case insensitive)
    matches = movies[movies['title'].str.lower().str.contains(movie)]

    if matches.empty:
        print("❌ Movie not found")
        return

    # Show top matches
    print("\n🔍 Matching movies:\n")
    for i, title in enumerate(matches['title'].head(5)):
        print(f"{i}: {title}")

    # User selects movie
    try:
        choice = int(input("\nEnter choice number: "))
        movie_index = matches.index[choice]
    except:
        print("❌ Invalid choice")
        return

    # Get similarity scores
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print(f"\n🎬 Recommendations:\n")

    for i in movies_list:
        print(movies.iloc[i[0]].title)


# -------------------------------
# ▶️ Run Program
# -------------------------------
if __name__ == "__main__":
    print("🎬 Movie Recommendation System")

    while True:
        movie = input("\nEnter movie name (or type 'exit'): ")

        if movie.lower() == 'exit':
            print("👋 Exiting...")
            break

        recommend(movie)