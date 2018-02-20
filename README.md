# spark_reco_demo
Spark Recommendations Demo for Brentwood AI Meetup 2018

To get started: run `./start_spark.sh` which will drop you into a spark shell

From there you'll need to start a master and slave with `./start_master_and_slave.sh`, Note that this also installs pip and numpy which is needed for ALS.

Then you're ready to test with `./test_spark.sh`, this will compute an approximation of pi using Spark

Finally you can run `spark-submit movielens.py`

This will download the small movielens dataset and run through a few training runs. Shouldn't take more than a few minutes on a modern machine.

Most of the credit goes to: https://github.com/jadianes/spark-movie-lens/blob/master/notebooks/building-recommender.ipynb

Another great blog post on the topic is here, this one is especially cool because they pull in the movie cover images from IMDB:
https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea
