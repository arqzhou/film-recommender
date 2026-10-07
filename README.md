# Overview

This project uses a recommender system trained on the [MovieLens 100k Dataset](https://www.kaggle.com/datasets/grouplens/movielens-latest-small/data) to make suggestions for a user based on their ratings. It accomplishes this using SQLite, TensorFlow and NumPy to query the dataset and implement the collaborative filtering mechanism with matrix factorization.

## Jump to:
- [Background](#background)
- [Design Process](#design-process)
  - [Challenges and Solutions](#challenges-and-solutions)


Background

Design Process

- Challenges & Solutions
- Results
- Limitations
- Insights & What’s Next

How to Use this Program Yourself

# Background

In 2026, I found myself going to movie theatres more times than I ever had before to see movies like *Project Hail Mary*, *Spiderman: Brand New Day*, *The Odyssey*, *Obsession*, and more. One of the most rewarding parts of this experience was the opportunity to discuss the movies with many people around me, even well after seeing the movie, which got me thinking: *“There are so many good movies that I’ve been recommended to go see. I know I have some unique tastes, so is there a way I can make something to narrow down which of these movies I would most enjoy?”* 

Note: While MovieLens itself comes from a recommender system, I found that I have not watched/rated enough movies to get recommendations that truly feel personalized (it might also be because I tend to rate almost every movie I watch at least a 3). This program has been trained to personalize much more aggressively for users like myself.

# Design Process

## Challenges and Solutions
