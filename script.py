import kagglehub
import pandas as pd
import sqlite3
import os

# Download latest version
path = kagglehub.dataset_download("grouplens/movielens-latest-small")

print("Path to dataset files:", path)

movies = pd.read_csv(f"{path}/movies.csv")
ratings = pd.read_csv(f"{path}/ratings.csv")
# print(movies.head())
# print(ratings.head())


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
    con.commit()
    con.close()
    
make_db()