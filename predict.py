import numpy as np
import tensorflow as tf
import sqlite3
import argparse

con = sqlite3.connect("movielens.db")
cur = con.cursor()

data = np.load("trained_weights.npz")

X_weights = data["X_weights"]
W_weights = data["W_weights"]
b_bias = data["b_bias"]
mean_ratings = data["mean_ratings"]
unique_mIds = data["unique_mIds"]

raw_preds = np.array(tf.linalg.matmul(X_weights, W_weights, transpose_b = True) + b_bias + mean_ratings[:, None])
preds = np.clip(raw_preds, 0.5, 5)

test_preds = preds[data["test_movie_indices"], data["test_user_indices"]]
rmse = np.sqrt(np.mean((test_preds - data["test_true_ratings"]) ** 2))
print(f"Model Test RMSE:    {rmse:.4f}")

baseline_rmse = np.sqrt(np.mean((mean_ratings[data["test_movie_indices"]] - data["test_true_ratings"]) ** 2))
print(f"Baseline Test RMSE: {baseline_rmse:.4f}")

# For debugging purposes
def find_user_top_movies(userId, n):
    search_query = """
        SELECT m.movieId, m.title, r.rating
        FROM movies m
        INNER JOIN ratings r
            ON m.movieId = r.movieId AND r.userId = ?
        ORDER BY r.rating DESC
        LIMIT ?
    ;
    """
    cur.execute(search_query, (userId, n))
    rows = cur.fetchall()

    print(f"\nTop {len(rows)} ratings for User {userId}:")
    for row in rows:
        print(f"  ({row[2]:.1f}, '{row[1]}')")


def get_recommendations(userId, n):
    # Find n popular movies (20+ ratings) that have the highest average rating.
    recommendations_query = """
        SELECT m.movieId, m.title, pop.avg_rating, pop.num_ratings
        FROM movies m
        INNER JOIN (
            SELECT movieId, AVG(rating) AS avg_rating, COUNT(*) AS num_ratings
            FROM ratings
            GROUP BY movieId
            HAVING COUNT(*) > 20
        ) pop ON m.movieId = pop.movieId
        WHERE m.movieId NOT IN (
            SELECT movieId
            FROM ratings
            WHERE userId = ?
        )
        ORDER BY pop.avg_rating DESC
        LIMIT ?;
    """
    cur.execute(recommendations_query, (userId, n))
    rows = cur.fetchall()

    print(f"\nTop {len(rows)} baseline recommendations for User {userId}:")
    for row in rows:
        print(f"  ({row[2]:.2f} avg, {row[3]} reviews, '{row[1]}')")
    return(rows)

def personal_preds(preds, userId, unique_mIds, X_weights, W_weights, n):
    user_index = int(userId) - 1
    pers_preds = preds[:, user_index]

    # Finds all movies that a parameterized user has not rated.
    unrated_query = """
        SELECT m.movieId, m.title
        FROM movies m
        INNER JOIN (
            SELECT movieId
            FROM ratings
            GROUP BY movieId
            HAVING COUNT(*) > 20
            ) pop
            ON m.movieId = pop.movieId
        LEFT JOIN ratings u
            ON m.movieId = u.movieId AND u.userId = ?
        WHERE u.movieId IS NULL;
    """
    cur.execute(unrated_query, (userId,))
    unrated_movie_data = cur.fetchall()
    unrated_ids, unrated_titles = zip(*unrated_movie_data)
    indexed_unrated_ids = np.searchsorted(unique_mIds, unrated_ids)

    user_vec = W_weights[user_index]
    user_norm = np.linalg.norm(user_vec)

    if user_norm > 0:
        movie_norms = np.linalg.norm(X_weights, axis=1)
        # Avoid division by zero for unrated/zero-norm movies
        safe_movie_norms = np.where(movie_norms == 0, 1.0, movie_norms)

        # Cosine similarity measures directional alignment in [-1.0, 1.0]
        cosine_sim = np.dot(X_weights, user_vec) / (safe_movie_norms * user_norm)

    unrated_scores = cosine_sim[indexed_unrated_ids]
    unrated_preds = pers_preds[indexed_unrated_ids]

    # Eliminate all movies where the user is predicted to have rated the movie below 4.5/5.
    quality_threshold = 4.5
    qualifying_mask = unrated_preds >= quality_threshold

    if np.sum(qualifying_mask) < n:
        top_fallback_indices = np.argsort(unrated_preds)[::-1][:50]
        qualifying_mask = np.zeros_like(unrated_preds, dtype = bool)
        qualifying_mask[top_fallback_indices] = True

    candidate_affs = unrated_scores[qualifying_mask]
    candidate_preds = unrated_preds[qualifying_mask]
    candidate_titles = np.array(unrated_titles)[qualifying_mask]

    # Decide the final n movies based on cosine similarity since deciding between clipped ratings is normally arbitrary.
    top_affs_indices = np.argsort(candidate_affs)[::-1][:n]

    final_titles = candidate_titles[top_affs_indices]
    final_affs = candidate_affs[top_affs_indices]
    final_preds = candidate_preds[top_affs_indices]

    dtype = np.dtype([('affinity', 'f4'), ('prediction', 'f4'), ('title', 'U100')])

    top_recommendations = np.empty(n, dtype=dtype)
    top_recommendations['affinity'] = final_affs
    top_recommendations['prediction'] = final_preds
    top_recommendations['title'] = final_titles

    print(f"\nTop {len(top_recommendations)} personalized recommendations for User {userId}:")
    for row in top_recommendations:
        print(
            f"  (Affinity: {row['affinity']:+.3f}, Pred: {row['prediction']:.2f},"
            f" '{row['title']}')"
        )
    return(top_recommendations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MovieLens Collaborative Filtering Inference"
    )
    parser.add_argument(
        "--user",
        type=str,
        default="611",
        help="Target userId to recommend movies for"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of Recommendations"
    )
    args = parser.parse_args()

    target_user = args.user
    n = args.top_n
    find_user_top_movies(target_user, n=30)
    get_recommendations(target_user, n=n)
    personal_preds(preds, target_user, unique_mIds, X_weights, W_weights, n=n)

    con.close()


