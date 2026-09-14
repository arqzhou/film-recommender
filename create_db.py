import kagglehub
import pandas as pd
import sqlite3
import os

# Download latest version
path = kagglehub.dataset_download("grouplens/movielens-latest-small")

print("Path to dataset files:", path)

movies = pd.read_csv(f"{path}/movies.csv")
ratings = pd.read_csv(f"{path}/ratings.csv")
#print(movies.head())
print(ratings.head())


def make_db():
    if os.path.exists("movielens.db"):
        os.remove("movielens.db")
        print("removed movielens.db for fresh start")
    con = sqlite3.connect("movielens.db")

    cur = con.cursor()
    cur.execute("PRAGMA foreign_keys = ON")
    cur.execute("""CREATE TABLE IF NOT EXISTS movies (
        movieId INT PRIMARY KEY,
        title VARCHAR NOT NULL,
        genres VARCHAR NOT NULL
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS ratings (
        userId INT NOT NULL,
        movieId INT NOT NULL,
        rating FLOAT NOT NULL,
        timestamp INT NOT NULL,
        PRIMARY KEY (userId, movieId),
        CONSTRAINT fk_movieId
            FOREIGN KEY (movieId) REFERENCES movies(movieId)
        ON DELETE CASCADE
    );""")
    #cur.executemany("INSERT INTO trips (trip_id, start_date, feed_timestamp, schedule_relationship, route_id, direction, vehicle_id) VALUES (?, ?, ?, ?, ?, ?, ?)", trips_data)
    print(movies.shape)
    cur.executemany("INSERT INTO movies (movieId, title, genres) VALUES (?, ?, ?)", movies.values.tolist())
    cur.executemany("INSERT INTO ratings (userId, movieId, rating, timestamp) VALUES (?, ?, ?, ?)", ratings.values.tolist())

    cur.execute("""
    """)

#     cur.execute("""
#         SELECT m.*, r.movie_count
#         FROM movies m
#         JOIN (
#             SELECT
#                 movieId,
#                 COUNT(*) AS movie_count
#             FROM ratings
#             WHERE rating = 5
#             GROUP BY movieId
#             HAVING movie_count > 50
#         ) r
#         ON m.movieId = r.movieId
#         ORDER BY r.movie_count DESC
#         ;
# """)
#     rows = cur.fetchall()

#     for row in rows:     
#         print(row)

    con.commit()
    con.close()
    
make_db()