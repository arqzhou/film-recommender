import sqlite3

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
        HAVING num_ratings > 2
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