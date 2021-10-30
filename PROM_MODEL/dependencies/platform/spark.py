"""
spark.py
~~~~~~~~

Module containing helper function for use with Apache Spark
"""

import __main__
import argparse

from os import environ, listdir, path
import json
from pyspark import SparkFiles
from pyspark.sql import SparkSession

from dependencies.platform import logging


def start_spark(app_name='my_spark_app', master='local[*]', jar_packages=[],
                files=[], spark_config={}):
    """Start Spark session, get Spark logger and load config files.

    :param app_name: Name of Spark app.
    :param master: Cluster connection details (defaults to local[*]).
    :param jar_packages: List of Spark JAR package names.
    :param files: List of files to send to Spark cluster (master and
        workers).
    :param spark_config: Dictionary of config key-value pairs.
    :return: A tuple of references to the Spark session, logger and
        config dict (only if available).
    """

    # detect execution environment
    flag_repl = not (hasattr(__main__, '__file__'))
    flag_debug = 'DEBUG' in environ.keys()

    if not (flag_repl or flag_debug):
        # get Spark session factory
        spark_builder = (
            SparkSession
                .builder
                .appName(app_name)
                .enableHiveSupport())
    else:
        # get Spark session factory
        spark_builder = (
            SparkSession
                .builder
                .master(master)
                .appName(app_name)
                .enableHiveSupport())

        # create Spark JAR packages string
        spark_jars_packages = ','.join(list(jar_packages))
        spark_builder.config('spark.jars.packages', spark_jars_packages)

        spark_files = ','.join(list(files))
        spark_builder.config('spark.files', spark_files)

        # add other config params
        for key, val in spark_config.items():
            spark_builder.config(key, val)

    # create session and retrieve Spark logger object
    spark_sess = spark_builder.getOrCreate()
    spark_logger = logging.Log4j(spark_sess)

    # parsing command line strings into Python dict.
    # Job param example: --job-args config_json=prod_job_demo_arima_config.json
    # reference: https://github.com/ekampf/PySpark-Boilerplate/blob/master/src/main.py
    job_args = dict()
    parser = argparse.ArgumentParser(description='Run a PySpark job')
    parser.add_argument('--job-args', nargs='*',
                        help="Extra arguments to send to the PySpark job (example: --job-args config_json=configs/prod_job_demo_config.json fc_date=20200722")
    args = parser.parse_args()
    if args.job_args:
        job_args_tuples = [arg_str.split('=') for arg_str in args.job_args]
        print('job_args_tuples: %s' % job_args_tuples)
        job_args = {a[0]: a[1] for a in job_args_tuples}

    # get config file from --job-args
    spark_files_dir = environ.get("SPARK_YARN_STAGING_DIR")
    file_name = job_args.get('config_json')
    if spark_files_dir:
        path_to_config_file = path.join(spark_files_dir, file_name)
    else:
        path_to_config_file = file_name
    spark_logger.info("path_to_config_file: " + path_to_config_file)
    config_file = spark_sess.sparkContext.textFile(path_to_config_file)
    config_dict = json.loads("".join(config_file.collect()))

    # merge args
    config_dict.update(job_args)
    spark_logger.info("config_dict: " + json.dumps(config_dict))

    return spark_sess, spark_logger, config_dict
