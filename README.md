# Overview

This project uses a recommender system trained on the [MovieLens 100k Dataset](https://www.kaggle.com/datasets/grouplens/movielens-latest-small/data) to make movie suggestions for a user based on their ratings. It accomplishes this by using SQLite, NumPy, and TensorFlow to query the dataset and implement the collaborative filtering mechanism with matrix factorization. The final model was able to achieve a RMSE value of 1.1097 compared to the baseline RMSE of 1.1713 and offer notably more personalized recommendations (see [Insights](#insights)) through a combination of a quality threshold and affinity score rankings.

## Jump to:
- [Background](#background)
- [How to Use this Program Yourself](#how-to-use-this-program-yourself)
- [Technical Design Process](#technical-design-process)
  - [Challenges and Solutions](#challenges-and-solutions)
      - [Cold Start Problem](#cold-start-problem)
      - [Standard Deviation](#standard-deviation)
      - [Data Leakage](#data-leakage)
      - [Affinity](#affinity)
      - [Choosing Lambda](#choosing-lambda)
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
   - fill in <userId> with any number 1 through 610. 611 will be your user ratings if you added them in step 2.
   - fill in <top_n> with the number of recommendations you want.
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

Based on this table, we can see that movies 2, 3, and 4 are similar because they share similar ratings between users who have watched multiple of them. We can also tell that these movies are different from movies 1 and 5 since they often have very different ratings. Because of this, we might infer that there is some hidden (latent) factor that differentiates these movies, such as genre, director, art style, etc.. This can be anything the model learns to pick up on.

For the example, let's say movies 2, 3, and 4 are animated movies and movies 1 and 5 are IMAX war movies. These hidden (latent) factors don't just affect movies, they also apply to users. Continuing the example, we can infer that users 1, 2, 5, and 6 prefer animated movies, and users 3 and 4 prefer the IMAX movies. This is how we crack the code of predicting how a user might rate a movie they've never seen before: if both the user and the movie heavily appeal to the same factor, then we can predict a positive rating.
To represent these preferences mathematically, we can create two new tables (matrices).
| movie | animated movies | IMAX war movie |
| --- | --- | --- |
| **Movie 1** | (1/10) | (8/10) |
| **Movie 2** | (9/10) | (0/10) |

| user | animated movies | IMAX war movie |
| --- | --- | --- |
| **User 4** | (2/10) | (9/10) |
| **User 5** | (8/10) | (3/10) |

To compare how movie/user preferences align, we can just take the sum of their products for each preference.
For movie 1 and user 4: $(\frac{1}{10})(\frac{2}{10})+(\frac{8}{10})(\frac{9}{10}) =  m_1 \cdot u_4 = \frac{74}{100}$

For movie 1 and user 5: $(\frac{1}{10})(\frac{8}{10})+(\frac{8}{10})(\frac{3}{10}) = m_1 \cdot u_5 =\frac{33}{100}$

As we can see, we are essentially taking the dot product of the user and movie k-vectors, and larger dot products show that the hidden features of the movie and user line up better.
On a single-movie single-user basis, this can be computed with a dot product, and on a large scale, this is simply matrix multiplication:
$(m\text{ movies} \times k\text{ hidden factors}) \times (k\text{ hidden factors} \times n\text{ users})$.

By multiplying our matrices together, we can predict which movies will be liked/disliked based on whether they appeal to similar preferences for every movie-user pair.

| movie | user 4 | user 5 |
| --- | --- | --- |
| **Movie 1** | (74/100) | (33/100) |
| **Movie 2** | (18/100) | (72/100) |

Note that the number of hidden factors is not limited to just 2. Because movie and user k-vectors are always k-long, the multiplication works whether k latent factors = 2 or 200. This current model uses ```k=15``` hidden factors.

Okay, now we can see how hidden factors can be useful, but how do we figure out each user/movie's values for each latent factor? This is where gradient descent comes in. Each user and movie is randomly assigned a value for every hidden factor to start. Then, we compare $m_i \cdot u_j$ to the actual rating (unrated movie-user pairs are skipped over). The total error (cost) is the sum of all these errors.

As you can imagine, the cost at the start is extremely high, but not all is lost. When calculating the cost, TensorFlow's ```tape.gradient``` automatically calculates the value of each derivative, which points every value in a user's or movie's k-vector in the direction it needs to change to lower the cost (especially if it heavily contributed to the cost function). ```apply_gradients``` brings these values slightly closer to their real preferences over time, and after 200 iterations, the error is much lower. Eventually, these values get to the point where more iterations doesn't really give us a better model (lower RMSE or good recommendations), and training should be stopped. Now, as long as you have every user's k-vector and every movie's k-vector, you can use those parameters to predict how any of those users will rate any of those movies.

There are a few more details in the training stage, such as the use of normalization, R_train masking, and the bias variable, and there is also big decisions made in ```predict.py```, but these will be discussed in the next section. Most of these decisions were made in reaction to a challenge that came up.

## Challenges and Solutions
### Cold Start Problem
The Cold Start Problem asks the question: how can we recommend good movies to watch if the user has barely rated any movies? For example, I haven't watched that many movies in this dataset, so I only rated 17 movies, compared to user 100, who had 148 ratings.

To help address this and acknowledge that certain movies are simply higher-rated than others, I implemented a version of mean normalization. This means for every movie-user pair where there was an actual rating, we subtract from it the average rating users gave that movie. Now, the model is trying to predict the difference from the average rating someone might rate a movie, rather than the rating of the movie itself. This is implemented here:
```
epsilon = .00001
mean_i = np.sum(Y * R_train, axis = 1) / (np.sum(R_train, axis = 1) + epsilon) # avoids the divide by 0 error
Y_norm = Y - mean_i[:, None] * R_train
```

This means that the baseline is now based on average movie ratings, rather than every movie being the same. If a user has never rated any movies, this model will just recommend the top-average-rated movies, and every additional rating a user has made gives the model a greater chance to personalize *from this baseline*. 

It is important to note that some niche movies are rated way less than popular movies in this dataset, so I implemented a requirement of more than 20 ratings to be recommended. Thus, these movies are still impactful for training the model, but this prevents an outlier movie with three 5-star ratings from being recommended over a movie with an average rating of 4.5 over 300 ratings. I decided this was the safer decision, as the model struggled to determine the latent factors of movies with only a couple of ratings, much like it is more difficult to predict movies for a user with fewer ratings. This was the first of several choices I made regarding the common popularity vs. personalization trade-off that I talk more about in [Insights](#insights).

### Standard Deviation
When testing an earlier version of the program, I was finding that my model was consistently predicting ratings of 7 to 9 for the top movies in a dataset with a range of ratings from 0.5 to 5 stars. Here, Claude recommended that I clip the values and try a variety of different measures, none of which worked to make the ratings more reasonable. Finally, I took one last look at the code with Gemini and figured out there were two factors in this below that were causing these extreme values.

```
X = tf.Variable(tf.random.normal((len(Y),k), dtype=tf.float64), name = "movie_features")
W = tf.Variable(tf.random.normal((len(Y[0]),k), dtype=tf.float64), name = "user_features")
b = tf.Variable(tf.random.normal((len(Y[0]),), dtype=tf.float64), name = "user_bias")
```

Firstly, the point of random initialization of these variables is so that there is are predicted differences we can compare to the actual difference ( $y_{norm_{ij}} - (X_i \cdot W^T_j + b_j)$ ). If $(X_i \cdot W^T_j + b_j)$ was all the same to start, we wouldn't know which features need to change for each movie/user. However, $b_j$ does not need to be randomly initialized since having distinct $X_i \cdot W^T_j$ will be enough; instead, $b_j$ should start from 0 and learn the general bias of the user (whether they tend to rate most movies higher than average or lower).

This itself is a small error, but it was compound by the second error, which is based on the default stddev value of ```tf.random.normal```. When stddev is not specified, this function initializes random variables based on a normal distribution with ```stddev = 1.0```. This means about 95% of all my values fall in between [-2, 2], and 99.7% fall in between [-3, 3], which was way too wide of a range. Immediately, this meant that it was quite normal for $b_j$ to get initialized to +2 or -2, which meant that the model predicted the user to rate everything 2 stars lower or higher than average, a substantial amount that was difficult to come back from.

This was made worse when calculating $X_{i} \cdot W^T_j$. If $X_{ik}$ and $W_{jk}$ happened to be initialized at 2, then the product would be 4 and heavily push the prediction positive. While the dot product should still average to 0, it only takes one large unbalanced variable to skew the prediction. This, combined with a possible +2 bias would give a +6 predicted *differential*. If that movie had an average rating of 4, the predicted rating would explode up to 10. To address this explosion, I explicitly set the ```stddev = .01``` and initialized b with ```tf.zeros```. Now, the initial values will start small and gradually grow, if needed, to make better predictions, rather than having very incorrect predictions and trying to reel them back in.
```
X = tf.Variable(tf.random.normal((len(Y),k), stddev = .01, dtype=tf.float64), name = "movie_features")
W = tf.Variable(tf.random.normal((len(Y[0]),k), stddev = .01, dtype=tf.float64), name = "user_features")
b = tf.Variable(tf.zeros((len(Y[0]),), dtype=tf.float64), name = "user_bias")
```

### Data Leakage
After addressing these bugs with the standard deviation, I decided to calculate the Root Mean Squared Error (RMSE) to objectively determine whether my model was better at predicting a user's rating of a movie than just assuming each user would give the movie's average rating. At first, I was excited that my RMSE seemed much lower than the baseline (```0.287``` vs ```0.87```). However, I later realized that this number was actually too low, and it was because I had forgotten to split out a test section. The RMSE was that low simply because the model had trained on the entire dataset and memorized the ratings.

```
def manual_train_test_split(R):
    movie_indices, user_indices = np.where(R == 1)
    test_mask = np.zeros(len(movie_indices), dtype = bool)
    
    for user in np.unique(user_indices):
        this_user = np.where(user_indices == user)[0]
        if len(this_user) >= 5:
            test_indices = np.random.choice(this_user, size = int(len(this_user) * .2), replace = False)
            test_mask[test_indices] = True
    
    R_train = R.copy()
    R_train[movie_indices[test_mask], user_indices[test_mask]] = 0
    return(R_train, movie_indices[test_mask], user_indices[test_mask])
```
To address this, I created a stratified train/test split, where every user with 5 or more ratings would have 20% of their ratings hidden from the training set. Now, the ```baseline RMSE: 1.1713```, and my actual ```RMSE = ~ 1.3```. However, now that the previous bugs were resolved, I was able to perform a lambda grid search at ```lambda = .1, .3, .5, .7, 1, 1.5, 2, 2.5, 3, 3.5, 5, 7.5, and 10``` and obtained these values:
| lambda | RMSE | less than baseline (1.1749) |
| --- | --- | --- |
| 0.1 | 1.3450 | N |
| 0.3 | 1.2837 | N |
| 0.5 | 1.2549 | N |
| 0.7 | 1.2411 | N |
| 1.0 | 1.2229 | N |
| 1.5 | 1.1860 | N |
| 2.0 | 1.1661 | Y |
| 2.5 | 1.1636 | Y |
| 3.0 | 1.1293 | Y |
| 3.5 | 1.1179 | Y |
| 5.0 | 1.0861 | Y |
| 7.5 | 1.0509 | Y |
| 10.0 | 1.0439 | Y |

However, it was not so simple as just choosing the lambda that resulted in the lowest RMSE. The recommendations still didn't feel great to me, so I first looked to see if there was a way I could further fine-tune the recommendations before deciding on a lambda.

### Affinity
Even after adjusting the standard deviation, I realized that I'd often get several prediction just barely over 5.0, like 5.04. This could be simply addressed with ```preds = np.clip(raw_preds, 0.5, 5)```, but this meant there was no way to distinguish between those 5-star-rated movies. Thus, the order of recommendations became extremely arbitrary, a 5.0+ predicted movie could be pushed out of the top 10 by other 5.0-rated films with no significant tiebreaker. Additionally, it also seemed like the difference between a 4.99 predicted film and a 5.0 predicted film was also quite arbitrary, especially for users with fewer ratings.

So, I decided to implement a tiebreaker that let more personalization show through in the recommendations using cosine similarity. Due to how my regularization term is calculated: ```reg_terms = lambda_*((tf.reduce_sum(X ** 2))) + lambda_*((tf.reduce_sum(W ** 2)))```, there is one penalty for having terms > 0, regardless of the number of ratings. However, the other half of the cost equation: ```tf.reduce_sum(((preds - Y_norm) ** 2) * R_train)``` calculates the cost for every rating. This means movies/users with lots of ratings will raise the cost more if they're inaccurate, so they are less impacted by the regularization term and can have greater vector magnitudes.

For example:
If a movie is rated 5 times, its cost would equal to the sum of those 5 differences + the square of its k_vector.
If a movie is rated 100 times, its cost would equal to the sum of those 100 differences + the square of its k_vector.

The square of the first scenario's k_vector factors much more in the cost equation than the second scenario, so that movie's k_vector is forced to have smaller terms, even if they would have had the same latent factors otherwise. While this is not a major flaw since it ensures some amount of "wisdom of the crowd" in the predicted ratings, it is not ideal for making the final decisions on recommendation ranking.

Luckily, there is a value that can be easily obtained from those values that ignores the number of ratings, and that is the cosine similarity angle, calculated by:

$$
cos(\theta) = \frac{m_i \cdot u_j}{\left\| m_i \right\|\times\left\| u_j \right\|}
$$
```
cosine_sim = np.dot(X_weights, user_vec) / (safe_movie_norms * user_norm)
```
The more similar the angle, the more the latent factors of the movie and the user line up, the greater the value of $cos(\theta)$ up to 1. Thus, the final recommendation system became a two-part system. Use matrix factorization to predict an overall quality of movies. At ```quality_threshold = 4.8```, the predicted rating difference between movies feels too arbitrary, so they are then ranked in descending order by $cos(\theta)$, or how well the movie aligns with the user's taste.

### Choosing Lambda
Adding cosine similarity didn't change the RMSE values for various lambda, since it is implemented after the predicted ratings, but it did change the feeling of the recommendations. Now that the recommendation system was complete, the lambda that performed the best seems to be ```lambda = 3.5```. The lower-value lambdas were still worse than the baseline, and the higher value lambdas didn't capture the personality of the user, recommending too many safe blockbuster picks.





## Example Results
4 kinds of users

## Insights
recs etc.

## Limitations
sparse, subjective final evaluation.

## Areas to Improve
make it easier to add ratings, or possibly make it autonomous.
mix in content-based filtering (genre tags are a part of the dataset)

   
