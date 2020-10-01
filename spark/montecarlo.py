from __future__ import print_function
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from operator import add
import random
import sys
import time

if __name__ == "__main__":
    """
    Usage: monte-carlo-portfolio.py [num_seeds] [scale_factor]
        - num_seeds: adjusts the data size
        - scale_factor: adjusts the computation time
    Based on example from https://cloud.google.com/solutions/monte-carlo-methods-with-hadoop-spark
    """
    num_seeds = 10000
    scale_factor = 1

    INVESTMENT_INIT = 100000  # starting amount
    INVESTMENT_ANN = 10000  # yearly new investment
    TERM = 30  # number of years
    MKT_AVG_RETURN = 0.11  # percentage
    MKT_STD_DEV = 0.01  # standard deviation

    num_executors = 5
    executor_image = "mesosphere/kubeflow:1.0.1-0.5.0-spark-3.0.0-horovod-0.19.5-tensorflow-2.2.0"

    conf = SparkConf()

    conf.setAll([
        ("spark.executor.instances", num_executors),
        ("spark.kubernetes.container.image", executor_image)])

    spark = SparkSession \
        .builder \
        .appName("Monte Carlo Portfolio") \
        .config(conf=conf) \
        .getOrCreate()

    def grow_one(seed):
        """
        Simulates the portfolio growth over TERM years based on
        one random seed.
        """
        random.seed(seed)
        portfolio_value = INVESTMENT_INIT
        for i in range(TERM):
            growth = random.normalvariate(MKT_AVG_RETURN, MKT_STD_DEV)
            portfolio_value += portfolio_value * growth + INVESTMENT_ANN
        return portfolio_value

    def grow(seed):
        """
        Adds an additional loop of size scale_factor inside the
        RDD map function.
        """
        portfolio_values = [grow_one(seed + i) for i in range(scale_factor)]
        avg = sum(portfolio_values) / len(portfolio_values)
        return avg

    sc = spark.sparkContext
    seeds = sc.parallelize([time.time() + i for i in range(num_seeds)])
    results = seeds.map(grow)
    sum = results.reduce(add)
    print('Average return: {}'.format(sum / num_seeds))
    print('Expected return: ~4.27 Million')

    print('Job completed successfully')

    spark.stop()
