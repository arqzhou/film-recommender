import sqlite3

con = sqlite3.connect("movielens.db")
cur = con.cursor()

cur.execute("""
    SELECT r.userId
    FROM ratings r
    GROUP BY r.userId
    ;
""")

users = cur.fetchall()
print(users[:10])

unrated_query = """
    SELECT m.*
    FROM movies m
    LEFT JOIN ratings r
        ON m.movieId = r.movieId AND r.userId = ?
    WHERE r.movieId IS NULL
    ;
"""

for user in users[:10]:
        cur.execute(unrated_query, user)
        unrated_rows = cur.fetchall()
        for unrated_row in unrated_rows[:10]:
            print(unrated_row)

print(len(users))

# rows = cur.fetchall()

# for row in rows[:10]:
#     print(row)