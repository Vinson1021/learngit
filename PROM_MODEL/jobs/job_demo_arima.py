"""
job_demo2.py
~~~~~~~~~~

This example script can be executed as follows,

    $SPARK_HOME/bin/spark-submit \
    --master local[5] \
    --py-files packages.zip \
    --files configs/dev_job_demo2_config.json \
    jobs/job_demo2.py

"""

from pyspark.sql import Row
from pyspark.sql.functions import col, concat_ws, lit

from dependencies.platform.spark import start_spark
from dependencies.algorithm.arima_demo import  ArimaModelDemo


#############################################
# main function
#############################################
def main():
    """Main script definition.

    :return: None
    """
    # start Spark application and get Spark session, logger and config
    spark, log, config = start_spark(
        app_name='my_job_demo2',
        files=['configs/prod_job_demo_arima_config.json'])

    # log that main spark job is starting
    log.warn('job_demo2 is up-and-running')

    # execute spark pipeline
    data = extract_data(spark)
    data_transformed = transform_data(data)
    write_data(data_transformed, config)

    # log the success and terminate Spark application
    log.warn('job_demo is finished')
    spark.stop()
    return None

#############################################
# extract data 
#############################################
def extract_data(spark):
    """Load data from Parquet file format.

    """
    _rdd = spark.sparkContext.parallelize([[1, 2, 3, 4, 5,6,7,8],
                   [11,12,13,14,15,16,17,18],
                   [21,22,23,24,25,26,27,28],
                   [31,32,33,34,35,36,37,38],
                   [41,42,43,44,45,46,47,48],
                   [51,52,53,54,55,56,57,58],
                   [61,62,63,64,65,66,67,68],
                   [71,72,73,74,75,76,77,78]]).repartition(2)
    return _rdd

#############################################
# data processing 
#############################################
def transform_data(_rdd):
    """Process original rdd.
    """
    # create Arima object
    arima_demo = ArimaModelDemo()

    result = _rdd.map(arima_demo.forecast)
    
    return result

#############################################
# output data
#############################################
def write_data(_rdd, config):
    """Collect data locally and write to CSV.

    :param df: rdd to print.
    :return: None
    """
    _rdd.saveAsTextFile(config['result_path'] + '/fc_dt=' + config['fc_dt'])
    return None


#############################################
# entry point for PySpark application
#############################################
if __name__ == '__main__':
    main()