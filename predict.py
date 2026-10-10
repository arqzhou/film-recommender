import numpy as np
import tensorflow as tf
import sqlite3
import argparse

np.random.seed(1)
tf.random.set_seed(1)

con = sqlite3.connect("movielens.db")
cur = con.cursor()

data = np.load("trained_weights.npz")

X_weights = data["X_weights"]
W_weights = data["W_weights"]
G_weights = data["G_weights"]
mg = data["mg"]
b_bias = data["b_bias"]
mean_ratings = data["mean_ratings"]
unique_mIds = data["unique_mIds"]

X_comb = np.array(X_weights + tf.linalg.matmul(mg, G_weights))
raw_preds = np.array(tf.linalg.matmul(X_comb, W_weights, transpose_b = True) + b_bias + mean_ratings[:, None])
preds = np.clip(raw_preds, 0.5, 5)

test_preds = preds[data["test_movie_indices"], data["test_user_indices"]]
rmse = np.sqrt(np.mean((test_preds - data["test_true_ratings"]) ** 2))
print(f"Model Test RMSE:    {rmse:.4f}")

baseline_rmse = np.sqrt(np.mean((mean_ratings[data["test_movie_indices"]] - data["test_true_ratings"]) ** 2))
print(f"Baseline Test RMSE: {baseline_rmse:.4f}")

# For debugging purposes
def find_user_top_movies(userId, n):
    search_query = """
        SELECT m.movieId, m.title, m.genres, r.rating
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
        print(f"  ({row[3]:.1f}, '{row[1]}', Genres: '{row[2]}'))")

    genre_count = {}
    for row in rows:
        for genre in row[2].split('|'):
            if genre in genre_count:
                genre_count[genre] += 1
            else:
                genre_count[genre] = 1
    sorted_genre_count = dict(sorted(genre_count.items(), key=lambda item: item[1], reverse=True))

    print("Top Genres:")
    for genre, count in list(sorted_genre_count.items())[:5]:
        print(f"{genre}: {count/len(rows):.2%}")


    return(False if len(rows) == 0 else True, sorted_genre_count)


def get_recommendations(userId, n):
    # Find n popular movies (20+ ratings) that have the highest average rating.
    recommendations_query = """
        SELECT m.movieId, m.title, m.genres, pop.avg_rating, pop.num_ratings
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
        print(f"  ({row[3]:.2f} avg, {row[4]} reviews, '{row[1]}', Genres: '{row[2]}')")
    return(rows)

def personal_preds(preds, userId, unique_mIds, X_comb, W_weights, sorted_genre_count, n):
    user_index = int(userId) - 1
    pers_preds = preds[:, user_index]

    # Finds all movies that a parameterized user has not rated.
    unrated_query = """
        SELECT m.movieId, m.title, m.genres
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
    unrated_ids, unrated_titles, unrated_genres = zip(*unrated_movie_data)
    indexed_unrated_ids = np.searchsorted(unique_mIds, unrated_ids)

    user_vec = W_weights[user_index]
    user_norm = np.linalg.norm(user_vec)

    if user_norm > 0:
        movie_norms = np.linalg.norm(X_comb, axis=1)
        # Avoid division by zero for unrated/zero-norm movies
        safe_movie_norms = np.where(movie_norms == 0, 1.0, movie_norms)

        # Cosine similarity measures directional alignment in [-1.0, 1.0]
        cosine_sim = np.dot(X_comb, user_vec) / (safe_movie_norms * user_norm)
    
    unrated_scores = cosine_sim[indexed_unrated_ids]
    unrated_preds = pers_preds[indexed_unrated_ids]
    unrated_blends = (unrated_preds/10) + 1 * (unrated_scores)

    # Eliminate all movies where the user is predicted to have rated the movie below 4.5/5.
    quality_threshold = 4.5
    qualifying_mask = unrated_preds >= quality_threshold

    if np.sum(qualifying_mask) < n:
        top_fallback_indices = np.argsort(unrated_preds)[::-1][:50]
        qualifying_mask = np.zeros_like(unrated_preds, dtype = bool)
        qualifying_mask[top_fallback_indices] = True

    candidate_affs = unrated_scores[qualifying_mask]
    candidate_preds = unrated_preds[qualifying_mask]
    candidate_genres = np.array(unrated_genres)[qualifying_mask]

    for i in range(len(candidate_affs)):
        num_matches = 0
        for genre in candidate_genres[i].split('|'):
            if genre in sorted_genre_count:
                candidate_affs[i] += (sorted_genre_count[genre] / (4 ** num_matches))
                num_matches += 1
            
        

    # candidate_blends = unrated_blends[qualifying_mask]
    candidate_titles = np.array(unrated_titles)[qualifying_mask]
    

    # Decide the final n movies based on cosine similarity since deciding between clipped ratings is normally arbitrary.
    # top_blends_indices = np.argsort(candidate_blends)[::-1][:n]
    top_affs_indices = np.argsort(candidate_affs)[::-1][:n]

    final_titles = candidate_titles[top_affs_indices]
    final_affs = candidate_affs[top_affs_indices]
    final_preds = candidate_preds[top_affs_indices]
    final_genres = candidate_genres[top_affs_indices]
    # final_blends = candidate_blends[top_blends_indices]

    # dtype = np.dtype([('blend', 'f4'), ('affinity', 'f4'), ('prediction', 'f4'), ('title', 'U100')])
    dtype = np.dtype([('affinity', 'f4'), ('prediction', 'f4'), ('title', 'U100'), ('genres', 'U100')])


    top_recommendations = np.empty(n, dtype=dtype)
    # top_recommendations['blend'] = final_blends
    top_recommendations['affinity'] = final_affs
    top_recommendations['prediction'] = final_preds
    top_recommendations['title'] = final_titles
    top_recommendations['genres'] = final_genres



    print(f"\nTop {len(top_recommendations)} personalized recommendations for User {userId}:")
    for row in top_recommendations:
        print(#  (Blend: {row['blend']:+.3f}, 
            f" Affinity: {row['affinity']:+.3f}, Pred: {row['prediction']:.2f},"
            f" '{row['title']}, Genres: '{row['genres']}')"
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
    rated_any_movies, sorted_genre_count = find_user_top_movies(target_user, n=30)
    get_recommendations(target_user, n=n)
    if rated_any_movies:
        personal_preds(preds, target_user, unique_mIds, X_comb, W_weights, sorted_genre_count, n=n,)
    else:
        print("This user has not rated any movies, so there is nothing to refine predictions from the baseline.")

    con.close()


