import kagglehub
import pandas as pd
# export KAGGLE_API_TOKEN=KGAT_ad027983caff1fe6007fb1edec17569d

# Download latest version
path = kagglehub.dataset_download("grouplens/movielens-latest-small")

print("Path to dataset files:", path)

movies = pd.read_csv(f"{path}/movies.csv")
print(movies.head())