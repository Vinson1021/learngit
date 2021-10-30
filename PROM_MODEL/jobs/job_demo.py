"""
job_demo.py
~~~~~~~~~~

This example script can be executed as follows,

    $SPARK_HOME/bin/spark-submit \
    --master local[5] \
    --py-files packages.zip \
    --files configs/etl_config.json \
    jobs/job_demo.py

"""
# import sys
# import os
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)


from pyspark.sql import Row
from pyspark.sql.functions import col, concat_ws, lit

from dependencies.platform.spark import start_spark

#############################################
# main function
#############################################
def main():
    """Main script definition.

    :return: None
    """
    # start Spark application and get Spark session, logger and config
    spark, log, config = start_spark(
        app_name='my_job_demo')

    # log that main spark job is starting
    log.warn('job_demo is up-and-running')

    # execute spark pipeline
    data = extract_data(spark, config)
    data_transformed = transform_data(data, config)
    write_data(data_transformed, config)

    # log the success and terminate Spark application
    log.warn('job_demo is finished')
    spark.stop()
    return None

#############################################
# extract data 
#############################################
def extract_data(spark, config):
    """Load data from Parquet file format.

    :param spark: Spark session object.
    :param config: Config params from json.
    :return: Spark DataFrame.
    """
    df = (
        spark
        .read
        .parquet(config['input_path']))

    return df


#############################################
# data processing 
#############################################
def transform_data(df, config):
    """Transform original dataset.

    :param df: Input DataFrame.
    :param config: Config params from json.
    :return: Transformed DataFrame.
    """
    df_transformed = (
        df
        .select(
            col('id'),
            concat_ws(
                ' ',
                col('first_name'),
                col('second_name')).alias('name'),
               (col('floor') * lit(config['steps_per_floor'])).alias('steps_to_desk')))

    return df_transformed


#############################################
# output data
#############################################
def write_data(df, config):
    """Collect data locally and write to CSV.

    :param df: DataFrame to print.
    :return: None
    """
    (df
     .coalesce(1)
     .write
     .csv(config['result_path'], mode='overwrite', header=True))
    return None


#############################################
# entry point for PySpark application
#############################################
if __name__ == '__main__':
    main()

