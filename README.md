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

| movie | user 1 | user 2 | user 3 | user 4 | user 5 | user 6 |
| --- | --- | --- | --- | --- | --- | --- |
| **Movie 1** | .5 | ? | 5 | 4.5 | 1 | 2 |
| **Movie 2** | 5 | 3.5 | 2 | ? | 5 | 4 |
| **Movie 3** | ? | ? | 1.5 | ? | ? | 5 |
| **Movie 4** | 4 | 5 | 1 | .5 | 4 | ? |
| **Movie 5** | ? | 1 | ? | 4 | ? | ? |

Based on this table, we can see that movies 2, 3, and 4 are similar because they share similar ratings between users who have watched multiple of them. We can also tell that these movies are different from movies 1 and 5 since they often have very different ratings. Because of this, we might infer that there is some hidden (latent) factor that differentiates these movies, such as genre, director, art style, etc..

For the example, let's say movies 2, 3, and 4 are animated movies and movies 1 and 5 are IMAX war movies. These hidden (latent) factors don't just apply to movies, they also apply to users. Continuing the example, we can infer that users 1, 2, 5, and 6 prefer animated movies, and users 3 and 4 prefer the IMAX movies. This is how we crack the code of predicting how a user might rate a movie they've never seen before: if both the user and the movie heavily appeal to the same factor, then we can predict a positive rating.
To represent these preferences mathematically, we can create two new tables (matrices).
| movie | animated movies | IMAX war movie |
| --- | --- | --- |
| **Movie 1** | (1/10) | (8/10) |
| **Movie 2** | (9/10) | (0/10) |

| user | animated movies | IMAX war movie |
| --- | --- | --- |
| **User 4** | (2/10) | (9/10) |
| **User 5** | (8/10) | (3/10) |

To compare how movie/user preferences align, we can just take the sum of their products for each preference. So for movie 1 and user 4, we'd get (1/10)(2/10) + (8/10)(9/10) = (74/100), which is a lot better of a match than movie 1 and user 5 (1/10)(9/10) + (8/10)(3/10) = (33/100). By multiplying our matrices together, we can predict which movies will be liked/disliked based on whether they appeal to similar preferences.

| movie | user 4 | user 5 |
| --- | --- | --- |
| **Movie 1** | (74/100) | (33/100) |
| **Movie 2** | (18/100) | (72/100) |

Note that the number of hidden factors is not limited to just 2. The math works whether k latent factors = 2 or 200. This current model uses 15 hidden factors.

Okay, now we can see how hidden factors can be useful, but how do we figure out each user/movie's values for each latent factor? This is where gradient descent comes in. Each user and movie is randomly assigned a value for every hidden factor to start. Then, every actual movie-user rating pair is multiplied together to form a prediction, whose difference from the actual rating is called an error. The total error is the sum of all these errors. As you can imagine, the total error at the start is extremely high, but not all is lost. When calculating the cost, TensorFlow's ```tape.gradient``` automatically calculates the value of each derivative, which points every value in the direction it needs to change (especially if it contributed to the cost function). ```apply_gradients``` brings these values slightly closer to their real preferences over time, and after 200 iterations, the error is a lot lower. Eventually, these values get to the point where more iterations doesn't really give us a better model (lower RMSE or good recommendations), and training should be stopped.

## Challenges and Solutions

## Example Results

## Insights

## Limitations

## Areas to Improve


   
