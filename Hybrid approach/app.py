import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, export_text
import shap

# Load datasets
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
tags = pd.read_csv('tags.csv')

# Step 1: Preprocess Data
# Combine tags and genres for content-based filtering
tags_agg = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
movies = movies.merge(tags_agg, on='movieId', how='left')
movies['tag'] = movies['tag'].fillna('')  # Fill NaN tags
movies['content'] = movies['genres'] + ' ' + movies['tag']  # Combine genres and tags
alpha =0.7

# Step 2: Build Collaborative Filtering Model
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Train SVD collaborative filtering model
cf_model = SVD()
cf_model.fit(trainset)

# Step 3: Build Content-Based Similarity Matrix
vectorizer = CountVectorizer(stop_words='english')

content_matrix = vectorizer.fit_transform(movies['content'])
content_similarity = cosine_similarity(content_matrix, content_matrix)

# Map movie IDs to indices
movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movies['movieId'])}

# Step 4: Hybrid Recommendation System
def hybrid_recommend(user_id, top_n=5, alpha=0.7):
    """
    Recommend top N movies based on a hybrid of collaborative filtering and content similarity.
    """
    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].unique()
    recommendations = []

    for movie_id in movies['movieId'].unique():
        if movie_id not in rated_movies:
            # Collaborative filtering prediction
            cf_pred = cf_model.predict(user_id, movie_id).est

            # Content-based similarity score
            movie_idx = movie_id_to_index[movie_id]
            unrated_indices = [movie_id_to_index[mid] for mid in rated_movies if mid in movie_id_to_index]
            content_pred = np.mean(content_similarity[movie_idx][unrated_indices]) if unrated_indices else 0

            # Hybrid score
            hybrid_score = alpha * cf_pred + (1 - alpha) * content_pred
            recommendations.append((movie_id, hybrid_score))

    # Sort and return top recommendations
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
    return [(movies[movies['movieId'] == movie_id]['title'].values[0], score,movie_id) for movie_id, score in recommendations]

# Step 5: Explainability
def explain_hybrid_recommendations(user_id, top_movies, rated_movies, alpha=0.7):
    """
    Explain the hybrid recommendations by breaking down scores into components.
    """
    for title, score , _ in top_movies:
        movie_id = movies[movies['title'] == title]['movieId'].values[0]
        cf_score = cf_model.predict(user_id, movie_id).est
        movie_idx = movie_id_to_index[movie_id]
        unrated_indices = [movie_id_to_index[mid] for mid in rated_movies if mid in movie_id_to_index]
        content_score = np.mean(content_similarity[movie_idx][unrated_indices]) if unrated_indices else 0

        print(f"Recommended: {title}")
        print(f" - Hybrid Score: {score:.2f}")
        print(f"   - Collaborative Filtering: {cf_score:.2f} ({alpha * 100:.0f}%)")
        print(f"   - Content-Based Filtering: {content_score:.2f} ({(1 - alpha) * 100:.0f}%)\n")




def decision_tree_explanation(movie_id):
    """
    Explain recommendations using decision tree rules for a specific movie.
    """
    movie_content = vectorizer.transform(movies[movies['movieId'] == movie_id]['content'])
    decision_path = tree_model.decision_path(movie_content)

    rules = export_text(tree_model, feature_names=vectorizer.get_feature_names_out())
    print(f"Decision Rules for Movie ID {movie_id} Recommendation:")
    print(rules)

    # Show which rules were applied for the specific movie
    print("\n--- Decision Path for the Selected Movie ---")
    for i, value in enumerate(decision_path.toarray()[0]):
        if value:
            print(f"Rule {i + 1}: {rules.splitlines()[i]}")



# Prepare data for decision tree explanation
ratings['liked'] = (ratings['rating'] >= 3.5).astype(int)
feature_data = pd.merge(ratings, movies, on='movieId')
X = vectorizer.transform(feature_data['content'])
y = feature_data['liked']

# Train decision tree
tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(X, y)

# Example: Recommend movies for a user
user_id = 1
rated_movies = ratings[ratings['userId'] == user_id]['movieId'].unique()
top_movies = hybrid_recommend(user_id, top_n=5)

print(f"Top recommendations for User {user_id}:")
for title, score,_ in top_movies:
    print(f"{title} (Hybrid Score: {score:.2f})")

# Explain recommendations
print("\n--- Explanation for Top Recommendations ---")
explain_hybrid_recommendations(user_id, top_movies, rated_movies)

print(top_movies)
# Decision Tree Explanation
print("\n--- Decision Tree Explanation ---")
decision_tree_explanation(top_movies[0][2])
