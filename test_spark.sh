#!/bin/bash
# $SPARK_HOME/bin/run-example SparkPi 100
spark-submit --class org.apache.spark.examples.SparkPi --master spark://spark:7077 $SPARK_HOME/examples/jars/spark-examples*.jar 100
