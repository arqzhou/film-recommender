import sqlite3
import numpy as np
import tensorflow as tf

con = sqlite3.connect("movielens.db")
cur = con.cursor()

def make_matrices():
    # movieId is non-contiguous, so creating a bidirectional mapping for new ids.
    movie_query = """
    SELECT movieId
    FROM movies
    ORDER BY movieId
    ;
    """
    cur.execute(movie_query)
    all_movieIds = cur.fetchall()
    print(all_movieIds[:5])
    unique_mIds = np.array(all_movieIds).squeeze()
    print(len(unique_mIds))

    rating_query = """
    SELECT userId, movieId, rating
    FROM ratings
    ORDER BY userId, movieId
    ;
    """
    cur.execute(rating_query)
    rating_data = cur.fetchall()
    

    # Vectorized way to add all the data into the matrix.
    m_userIds, m_movieIds, m_ratings = zip(*rating_data)
    a_userIds = tuple(x - 1 for x in m_userIds)

    # Assigns the noncontiguous values to a contiguous index from 0 to N - 1, where N = # of unique movie Ids.
    m_movieIds = np.searchsorted(unique_mIds, m_movieIds)

    Y_matrix = np.zeros((len(unique_mIds), len(np.unique(a_userIds))))
    R_matrix = np.zeros((len(unique_mIds), len(np.unique(a_userIds))))

    Y_matrix[m_movieIds, a_userIds] = m_ratings
    R_matrix[m_movieIds, a_userIds] = 1
    print(Y_matrix[:7, :5])
    print(R_matrix[:3, :5])

    return(Y_matrix, R_matrix, unique_mIds)

def manual_train_test_split(R):
    movie_indices, user_indices = np.where(R == 1)
    print(movie_indices[:10])
    print(user_indices[:10])
    test_mask = np.zeros(len(movie_indices), dtype = bool)
    

    for user in np.unique(user_indices):
        this_user = np.where(user_indices == user)[0]
        if len(this_user) >= 5:
            test_indices = np.random.choice(this_user, size = int(len(this_user) * .2), replace = False)
            test_mask[test_indices] = True
    
    R_train = R.copy()
    R_train[movie_indices[test_mask], user_indices[test_mask]] = 0

    print(R_train[:10, :20])
    print(R[:10, :20])

    return(R_train, movie_indices[test_mask], user_indices[test_mask])

Y, R, unique_mIds = make_matrices()
R_train, test_movie_indices, test_user_indices = manual_train_test_split(R)

def cost_function(X, W, b, Y_norm, R_train, nm, nu, lambda_):
    preds = tf.linalg.matmul(X, W, transpose_b = True) + b
    squared_error = tf.reduce_sum(((preds - Y_norm) ** 2) * R_train)
    reg_terms = lambda_*((tf.reduce_sum(X ** 2))) + lambda_*((tf.reduce_sum(W ** 2)))

    return(squared_error + reg_terms)

def training_loop(Y, R_train):
    print(len(Y), len(Y[1]))
    X = tf.Variable(tf.random.normal((len(Y),25), stddev = .01, dtype=tf.float64), name = "movie_features")
    W = tf.Variable(tf.random.normal((len(Y[0]),25), stddev = .01, dtype=tf.float64), name = "user_features")
    b = tf.Variable(tf.zeros((len(Y[0]),), dtype=tf.float64), name = "user_bias")
    epsilon = .00001
    mean_i = np.sum(Y * R_train, axis = 1) / (np.sum(R_train, axis = 1) + epsilon) # avoids the divide by 0 error
    Y_norm = Y - mean_i[:, None] * R_train
    print(Y_norm[:5, :5])


    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-1)
    iterations = 200
    lambda_ = .1

    for iter in range(iterations):
        with tf.GradientTape() as tape:
            cost_value = cost_function(X, W, b, Y_norm, R_train, len(Y), len(Y[0]), lambda_)
        grads = tape.gradient(cost_value, [X, W, b])
        optimizer.apply_gradients(zip(grads, [X, W, b]))
        if iter % 10 == 0:
            print(cost_value, iter)

    return(X, W, b, mean_i)

X_weights, W_weights, b_bias, mean_ratings = training_loop(Y, R_train)

np.savez("trained_weights.npz", X_weights=X_weights.numpy(), W_weights=W_weights.numpy(), b_bias=b_bias.numpy(), mean_ratings=mean_ratings, unique_mIds=unique_mIds, test_movie_indices=test_movie_indices, test_user_indices=test_user_indices, test_true_ratings= Y[test_movie_indices, test_user_indices])
print("saved")



