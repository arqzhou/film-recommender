# Overview

This project uses a recommender system trained on the [MovieLens 100k Dataset](https://www.kaggle.com/datasets/grouplens/movielens-latest-small/data) to make movie suggestions for a user based on their ratings. It accomplishes this by using SQLite, NumPy, and TensorFlow to query the dataset and implement the collaborative filtering mechanism with matrix factorization. The final model was able to achieve a RMSE value of 1.1097 compared to the baseline RMSE of 1.1713 and offer notably more personalized recommendations (see [Insights](#insights)) with a combination of a quality threshold and affinity score rankings.

## Jump to:
- [Background](#background)
- [Design Process](#design-process)
  - [Challenges and Solutions](#challenges-and-solutions)
  - [Example Results](#example-results)
  - [Insights](#insights)
  - [Limitations](#limitations)
  - [Areas to Improve](#areas-to-improve)
- [How to Use this Program Yourself](#how-to-use-this-program-yourself)

# Background

In 2026, I found myself going to movie theatres more times than I ever had before to see movies like *Project Hail Mary*, *Spiderman: Brand New Day*, *The Odyssey*, *Obsession*, and more. One of the most rewarding parts of this experience was the opportunity to discuss the movies with many people around me, even well after seeing the movie, which got me thinking: *“There are so many good movies that I’ve been recommended to go see. I know I have some unique tastes, so is there a way I can make something to narrow down which of these movies I would most enjoy?”* 

Note: While MovieLens itself comes from a recommender system, I found that I have not watched/rated enough movies to get recommendations that truly feel personalized (it might also be because I tend to rate almost every movie I watch at least a 3). This program has been trained to personalize much more aggressively for users like myself.

# Design Process
Disclaimer: The code for this project was produced with the help of AI. However, I strongly believe in the importance of understanding the code I am using and have gone through any AI-generated code line-by-line. As a final check for understanding, everything I write in this README will be written by me as I walk you through the code and the choices that were made.

The project structure itself is straightforward.
1. ```create_db.py``` imports the data from Kaggle into a new file ```movielens.db```
2. ```train.py``` trains the model on data queried from ```movielens.db```
3. ```train.py``` saves its model parameters and other useful metrics on ```trained_weights.npz```
4. ```predict.py``` uses the data from ```trained_weights.npz``` to give baseline and personalized recommendations
5. (optional) ```write_movies.py``` imports ratings from ```my_ratings.txt``` into ```movielens.db``` as user 611
     - ```movies.txt``` contains all movies in the dataset that the user can rank from



## Challenges and Solutions

## Example Results

## Insights

## Limitations

## Areas to Improve

# How to Use this Program Yourself
1. Clone the repository. Make sure you have NumPy, TensorFlow, and Pandas installed.
   ```
   git clone https://github.com/arqzhou/film-recommender
   cd film-recommender
   pip install -r requirements.txt
   ```
2. Now, you can:
     a. Create your own ratings to see recommendations (move on to step 3)
     b. Train the model on different parameters such as lambda, lr, or iters (move on to step x)
     c. Test the model's results (move on to step y)
3. Creating your own ratings.
   - Copy the data from ```movies.txt``` and paste these into a Google or Excel Spreadsheet.
   - Insert one column to the very left, and rate movies you've watched on a scale of 0.5 to 5.0 stars.
   - Select all columns and turn your data into a table view. Filter to show only the rows that contain a rating.
   - Copy and paste these values into ```my_ratings.txt``` and save them.
   - Run ```python create_db.py``` to clean the database, followed by ```python write_movies.py``` to write your new values into ```movie_lens.db```.
   - Congratulations! Now your own ratings are a part of the dataset under the userId 611.
4. Training the model
   - If you wish to adjust the training of the model, you can adjust lambda
