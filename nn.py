import sqlite3
from bidict import bidict
import numpy as np

con = sqlite3.connect("movielens.db")
cur = con.cursor()


# Finds all the unique user IDs in the dataset and keeps them in users.
cur.execute("""
    SELECT r.userId
    FROM ratings r
    GROUP BY r.userId
    ;
""")

users = cur.fetchall()
print(users[:10])

# Finds all movies that a parameterized user(s) has not rated.
unrated_query = """
    SELECT m.*, r.userId, r.movieId
    FROM movies m
    LEFT JOIN ratings r
        ON m.movieId = r.movieId AND r.userId = ?
    WHERE r.movieId IS NULL
    ;
"""
# Printing for sanity check
for user in users[:3]:
        cur.execute(unrated_query, user)
        unrated_rows = cur.fetchall()
        for unrated_row in unrated_rows[:10]:
            print(unrated_row)

# Create popularity baseline if a new user hasn't rated anything.
baseline_query = """
    SELECT m.movieId, m.title, i.avg_rating, i.num_ratings
    FROM movies m
    INNER JOIN (
        SELECT movieId, AVG(rating) as avg_rating, COUNT(*) as num_ratings
        FROM ratings
        GROUP BY movieId
        HAVING num_ratings > 10
        ORDER BY avg_rating DESC) i
    ON m.movieId = i.movieId
    ;
"""
cur.execute(baseline_query)
avg_ratings = cur.fetchall()
print(avg_ratings[:10])

# Checks the amount of movies that would be kept based on rating threshold
# percentage_query = """
#     SELECT COUNT(*)
#     FROM (
#         SELECT movieId, AVG(rating) as avg_rating, COUNT(*) as num_ratings
#         FROM ratings
#         GROUP BY movieId
#         HAVING num_ratings > 2);
# """
# cur.execute(percentage_query)
# num_left = cur.fetchall()
# print(num_left)

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

# print(get_recommendations('10', 10))

def make_matrices():
    # movieId is non-contiguous, so creating a bidirectional mapping for new ids.
    movie_query = """
    SELECT movieId
    FROM movies
    ORDER BY movieId
    ;
    """
    cur.execute(movie_query)
    all_movieIds = cur.fetchall()
    print(all_movieIds[:5])
    unique_mIds = np.array(all_movieIds).squeeze()
    print(len(unique_mIds))

    rating_query = """
    SELECT userId, movieId, rating
    FROM ratings
    ORDER BY userId, movieId
    ;
    """
    cur.execute(rating_query)
    rating_data = cur.fetchall()
    print(len(rating_data))

    # Vectorized way to add all the data into the matrix.
    m_userIds, m_movieIds, m_ratings = zip(*rating_data)
    a_userIds = tuple(x - 1 for x in m_userIds)

    # Assigns the noncontiguous values to a contiguous index from 0 to N - 1, where N = # of unique movie Ids.
    m_movieIds = np.searchsorted(unique_mIds, m_movieIds)

    r_matrix = np.zeros((len(unique_mIds), len(np.unique(a_userIds))))
    b_matrix = np.zeros((len(unique_mIds), len(np.unique(a_userIds))))

    r_matrix[m_movieIds, a_userIds] = m_ratings
    b_matrix[m_movieIds, a_userIds] = 1
    print(r_matrix[:5, :5])
    print(b_matrix[:5, :5])

make_matrices()
