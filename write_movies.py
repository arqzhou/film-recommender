import sqlite3
import pandas as pd

con = sqlite3.connect("movielens.db")
cur = con.cursor()

# query = """
#     SELECT *
#     FROM movies
# """
# cur.execute(query)
# movie_list = cur.fetchall()

# print(type(movie_list))

# # This will create 'example.txt' or erase its current contents
# with open("movies.txt", "w", encoding="utf-8") as file:
#     for id, title, genres in movie_list:
#         file.writelines(f"{id}\t{title}\t{genres}\n")

def import_personal_ratings():
    my_ratings = pd.read_csv("my_ratings.txt", sep="\t", encoding="utf-8", header = None)
    insert_query = """
    INSERT INTO ratings
    (userId, movieId, rating, timestamp)
    VALUES (611, ?, ?, 1493846415)
    """
    print(my_ratings[0])
    formatted_ratings = my_ratings[[1,0]]

    cur.executemany(insert_query, formatted_ratings.values.tolist())

    print(formatted_ratings.head())
    con.commit()
    con.close()


import_personal_ratings()


