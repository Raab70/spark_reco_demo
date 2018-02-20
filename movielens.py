#!/usr/bin/env python
# Credit: https://github.com/jadianes/spark-movie-lens/blob/master/notebooks/building-recommender.ipynb
import os
import urllib
import zipfile
import math

from pyspark.mllib.recommendation import ALS
from pyspark.sql import SparkSession


if __name__ == '__main__':
    complete_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
    small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    datasets_path = os.path.join(os.getcwd(), 'datasets')
    try:
        os.makedirs(datasets_path)
    except OSError:
        pass
    small_dataset_path = os.path.join(datasets_path, 'ml-latest-small.zip')
    small_f = urllib.urlretrieve(small_dataset_url, small_dataset_path)
    with zipfile.ZipFile(small_dataset_path, "r") as z:
        z.extractall(datasets_path)

    spark = SparkSession \
        .builder \
        .appName("VideoAndRecircRecommendations") \
        .getOrCreate()
    sc = spark.sparkContext

    small_ratings_file = os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')

    small_ratings_raw_data = sc.textFile(small_ratings_file)
    small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]
    small_ratings_data = small_ratings_raw_data\
        .filter(lambda line: line != small_ratings_raw_data_header)\
        .map(lambda line: line.split(","))\
        .map(lambda tokens: (tokens[0],tokens[1],tokens[2]))\
        .cache()
    small_movies_file = os.path.join(datasets_path, 'ml-latest-small', 'movies.csv')

    small_movies_raw_data = sc.textFile(small_movies_file)
    small_movies_raw_data_header = small_movies_raw_data.take(1)[0]

    small_movies_data = small_movies_raw_data\
        .filter(lambda line: line != small_movies_raw_data_header)\
        .map(lambda line: line.split(","))\
        .map(lambda tokens: (tokens[0], tokens[1]))\
        .cache()

    # small_movies_data.take(3)
    training_RDD, validation_RDD, test_RDD = small_ratings_data\
        .randomSplit([6, 2, 2], seed=0L)
    validation_for_predict_RDD = validation_RDD\
        .map(lambda x: (x[0], x[1]))
    test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

    seed = 5L
    iterations = 10
    regularization_parameter = 0.1
    ranks = [4, 8, 12]
    errors = [0, 0, 0]
    err = 0
    tolerance = 0.02

    min_error = float('inf')
    best_rank = -1
    best_iteration = -1
    for rank in ranks:
        print "Beginning training for Rank %d" % rank
        model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                        lambda_=regularization_parameter)
        predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        errors[err] = error
        err += 1
        print 'For rank %s the RMSE is %s' % (rank, error)
        if error < min_error:
            min_error = error
            best_rank = rank

    print 'The best model was trained with rank %s' % best_rank
    model = ALS.train(
        training_RDD,
        best_rank,
        seed=42L,
        iterations=10,
        lambda_=0.1)
    predictions = model\
        .predictAll(test_for_predict_RDD)\
        .map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = test_RDD\
        .map(lambda r: ((int(r[0]), int(r[1])), float(r[2])))\
        .join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    print 'For testing data the RMSE is %s' % (error)
