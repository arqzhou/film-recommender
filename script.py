import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("grouplens/movielens-latest-small")

print("Path to dataset files:", path)

movies = pd.read_csv(f"{path}/movies.csv")
print(movies.head())