# Overview

This project uses a recommender system trained on the [MovieLens 100k Dataset](https://www.kaggle.com/datasets/grouplens/movielens-latest-small/data) to make movie suggestions for a user based on their ratings. It accomplishes this by using SQLite, NumPy, and TensorFlow to query the dataset and implement the collaborative filtering mechanism with matrix factorization. The final model was able to achieve a RMSE value of 1.1097 compared to the baseline RMSE of 1.1713 and offer notably more personalized recommendations (see [Insights](#insights)) with a combination of a quality threshold and affinity score rankings.

## Jump to:
- [Background](#background)
- [How to Use this Program Yourself](#how-to-use-this-program-yourself)
- [Technical Design Process](#technical-design-process)
  - [Challenges and Solutions](#challenges-and-solutions)
  - [Example Results](#example-results)
  - [Insights](#insights)
  - [Limitations](#limitations)
  - [Areas to Improve](#areas-to-improve)


# Background

In 2026, I found myself going to movie theatres more times than I ever had before to see movies like *Project Hail Mary*, *Spiderman: Brand New Day*, *The Odyssey*, *Obsession*, and more. One of the most rewarding parts of this experience was the opportunity to discuss the movies with many people around me, even well after seeing the movie, which got me thinking: *“There are so many good movies that I’ve been recommended to go see. I know I have some unique tastes, so is there a way I can make something to narrow down which of these movies I would most enjoy?”* 

Note: While MovieLens itself comes from a recommender system, I found that I have not watched/rated enough movies to get recommendations that truly feel personalized (it might also be because I tend to rate almost every movie I watch at least a 3). This program has been trained to personalize much more aggressively for users like myself.

# How to Use this Program Yourself
1. Clone the repository. Make sure you have NumPy, TensorFlow, and Pandas installed.
   ```
   git clone https://github.com/arqzhou/film-recommender
   cd film-recommender
   pip install -r requirements.txt
   ```
2. Now, you can:
   - Create your own ratings to see recommendations (move on to step 3)
   - Train the model on different parameters such as ```lambda```, ```lr```, ```iters```, or ```k``` (move straight to step 4)
   - Test the model's results (move straight to step 5)
4. Creating your own ratings.
   - Copy the data from ```movies.txt``` and paste these into a Google or Excel Spreadsheet.
   - Insert one column to the very left, and rate movies you've watched on a scale of 0.5 to 5.0 stars.
   - Select all columns and turn your data into a table view. Filter to show only the rows that contain a rating.
   - Copy and paste these values into ```my_ratings.txt``` and save them.
   - Run ```python create_db.py``` to clean the database, followed by ```python write_movies.py``` to write your new values into ```movie_lens.db```.
   - Congratulations! Now your own ratings are a part of the dataset under the userId 611.
5. Training the model.
   - If you wish to adjust the training of the model, you can adjust the parameters at the top of the ```training_loop()``` function.
   ```
   def training_loop(Y, R_train):
    # Adjust parameters here
    iters = 200
    lambda_ = 3.5
    k = 15
    lr = 1e-1
   ```
   - Then, go ahead and train the model by running ```python train.py```. This will save your new weights to ```trained_weights.npz```.
6. Test the model.
   - To test the model, simply run the command below.
   - <userId> can be any number 1 through 610. 611 will be your user ratings if you added them in step 2.
   - <top_n> is the number of recommendations you want.
   - This code will return up to the top 30 highest rated movies by the user, along with baseline recommendations (top n movies by average rating) and personalized recommendations (top n movies decided by collaborative filtering model and affinity)
   ```
   python predict.py --user <userId> --top_n <top_n>
   ```

   Below is an example output for my 16 ratings as user 611.

   ```
   Top 16 ratings for User 611:
   (5.0, 'The Blue Planet (2001)')
   (5.0, 'Kung Fu Panda 3 (2016)')
   (5.0, 'Your Name. (2016)')
   (5.0, 'Moana (2016)')
   (5.0, 'Planet Earth II (2016)')
   (5.0, 'Blue Planet II (2017)')
   (4.5, 'Zootopia (2016)')
   (4.5, 'The Man Who Knew Infinity (2016)')
   (4.5, 'Hidden Figures (2016)')
   (4.0, 'Ice Age: The Great Egg-Scapade (2016)')
   (4.0, 'Sully (2016)')
   (3.5, 'The Angry Birds Movie (2016)')
   (3.5, 'Cars 3 (2017)')
   (3.5, 'Incredibles 2 (2018)')
   (3.0, 'Finding Dory (2016)')
   (2.0, 'Storks (2016)')

   Top 10 baseline recommendations for User 611:
   (4.43 avg, 317 reviews, 'Shawshank Redemption, The (1994)')
   (4.33 avg, 27 reviews, 'Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)')
   (4.31 avg, 29 reviews, 'Philadelphia Story, The (1940)')
   (4.30 avg, 25 reviews, 'In the Name of the Father (1993)')
   (4.30 avg, 45 reviews, 'Lawrence of Arabia (1962)')
   (4.29 avg, 29 reviews, 'Hoop Dreams (1994)')
   (4.29 avg, 192 reviews, 'Godfather, The (1972)')
   (4.29 avg, 26 reviews, 'Harold and Maude (1971)')
   (4.28 avg, 25 reviews, 'Logan (2017)')
   (4.27 avg, 218 reviews, 'Fight Club (1999)')

   Top 10 personalized recommendations for User 611:
   (Affinity: +0.625, Pred: 4.53, 'Princess Mononoke (Mononoke-hime) (1997)')
   (Affinity: +0.459, Pred: 4.74, 'Schindler's List (1993)')
   (Affinity: +0.415, Pred: 4.54, 'Glory (1989)')
   (Affinity: +0.414, Pred: 4.54, 'Shine (1996)')
   (Affinity: +0.405, Pred: 4.64, 'Like Water for Chocolate (Como agua para chocolate) (1992)')
   (Affinity: +0.404, Pred: 4.91, 'Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)')
   (Affinity: +0.403, Pred: 4.59, 'Guardians of the Galaxy (2014)')
   (Affinity: +0.400, Pred: 4.65, 'Princess Bride, The (1987)')
   (Affinity: +0.384, Pred: 4.65, 'Dark Knight, The (2008)')
   (Affinity: +0.373, Pred: 4.53, 'Inglourious Basterds (2009)')
   ```

# Technical Design Process
Disclaimer: The code for this project was produced with the help of AI. However, I strongly believe in the importance of understanding the code I am using and have gone through any AI-generated code line-by-line. As a final check for understanding, everything I write in this README will be written by me as I walk you through the code and the choices that were made.

To explain the main mathematical concept of the model, it's best to see an example table. 8 different users gave ratings for 5 movies, leaving them blanked if they haven't been watched.

| movie | user 1 | user 2 | user 3 | user 4 | user 5 | *user 6* | user 7 | user 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Movie 1** | .5 | ? | 5 | 4.5 | 1 | 2 | ? | ? |
| **Movie 2** | 5 | 3.5 | 2 | ? | 5 | 4 | 5 | 2 |
| **Movie 3** | ? | ? | 1.5 | ? | ? | 5 | ? | ? |
| **Movie 4** | 4 | 5 | 1 | .5 | 4 | ? | 4.5 | 3 |
| **Movie 5** | ? | 1 | ? | 4 | ? | ? | ? | 1 |

Based on this table, it might be possible to infer that movies 2, 3, and 4 are similar since users that have watched multiple of these have all rated them highly (1, 2, 5, and 7). For the same reason, movies 1 and 5 are likely similar, but they are also likely very different from the other 3 movies because the ratings tend to be quite disparate in users who have watched both groups. Thus, we can use information we've inferred about the user and about the movie to predict what a certain user might rate a certain movie. For example, we can likely infer that *user 6* will rate movie 4 highly but dislike movie 1.

## Challenges and Solutions

## Example Results

## Insights

## Limitations

## Areas to Improve


   
