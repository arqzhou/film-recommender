import numpy as np
import tensorflow as tf
import sqlite3

con = sqlite3.connect("movielens.db")
cur = con.cursor()

data = np.load("trained_weights.npz")

X_weights = data["X_weights"]
W_weights = data["W_weights"]
b_bias = data["b_bias"]
mean_ratings = data["mean_ratings"]
unique_mIds = data["unique_mIds"]

preds = np.array(tf.linalg.matmul(X_weights, W_weights, transpose_b = True) + b_bias + mean_ratings[:, None])

def get_recommendations(userId, n):
    recommendations_query = """
        SELECT f.movieId, f.title, f.avg_rating, f.num_ratings
        FROM (
            SELECT m.movieId, m.title, i.avg_rating, i.num_ratings
            FROM movies m
            INNER JOIN (
                SELECT movieId, AVG(rating) as avg_rating, COUNT(*) as num_ratings
                FROM ratings
                GROUP BY movieId
                HAVING num_ratings > 2
            ) i
            ON m.movieId = i.movieId
        ) f
        INNER JOIN (
            SELECT m.*, r.userId, r.movieId
                FROM movies m
                LEFT JOIN ratings r
                    ON m.movieId = r.movieId AND r.userId = ?
                WHERE r.movieId IS NULL
        ) u
        ON f.movieId = u.movieId
        ORDER BY f.avg_rating DESC
    ;
    """
    cur.execute(recommendations_query, (userId,))
    top_unrated_movies = cur.fetchall()
    return(top_unrated_movies[:n])

def personal_preds(preds, userId, n, unique_mIds):
    user_index = int(userId) - 1
    pers_preds = preds[:, user_index]
    # Finds all movies that a parameterized user(s) has not rated.
    unrated_query = """
        SELECT m.movieId, m.title
        FROM movies m
        LEFT JOIN ratings r
            ON m.movieId = r.movieId AND r.userId = ?
        WHERE r.movieId IS NULL
        ;
    """
    cur.execute(unrated_query, (userId,))
    unrated_movie_data = cur.fetchall()
    unrated_ids, unrated_titles = zip(*unrated_movie_data)
    print(unrated_titles[:10])
    indexed_unrated_ids = np.searchsorted(unique_mIds, unrated_ids)
    unrated_preds = pers_preds[indexed_unrated_ids]

    sorted_unrated_indices = np.argsort(unrated_preds)[::-1]
    top_preds = unrated_preds[sorted_unrated_indices][:n]
    top_titles = np.array(unrated_titles)[sorted_unrated_indices][:n]
    
    dtype = np.dtype([('prediction', 'f4'), ('title', 'U100')])

    top_recommendations = np.empty(top_preds.shape, dtype=dtype)
    top_recommendations['prediction'] = np.clip(top_preds, 0.5, 5)
    top_recommendations['title'] = top_titles

    return(top_recommendations)
    
print(get_recommendations('1', 10))
print(personal_preds(preds, '1', 10, unique_mIds))

print("\n-----------------------\n")

print(get_recommendations('50', 10))
print(personal_preds(preds, '50', 10, unique_mIds))

print("\n-----------------------\n")

print(get_recommendations('100', 10))
print(personal_preds(preds, '100', 10, unique_mIds))

print("\n-----------------------\n")

print(get_recommendations('611', 10))
print(personal_preds(preds, '611', 10, unique_mIds))

