"""
create_test_data.py
~~~~~~~~~~

This example script can be executed as follows,

    spark-submit \
    --master local[*] \
    --py-files packages.zip \
    --files configs/dev_create_test_data_config.json \
    jobs/create_test_data.py

"""
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


from pyspark.sql import Row
from pyspark.sql.functions import col, concat_ws, lit

from dependencies.platform.spark import start_spark


def create_test_data():
    """Create test data.

    This function creates both both pre- and post- transformation data
    saved as Parquet files in tests/test_data. This will be used for
    unit tests as well as to load as part of the example ETL job.
    :return: None
    """

      # start Spark application and get Spark session, logger and config
    spark, log, config = start_spark(
        app_name='create_test_data',
         files=[])

    # create example data from scratch
    local_records = [
        Row(id=1, first_name='Dan', second_name='Germain', floor=1),
        Row(id=2, first_name='Dan', second_name='Sommerville', floor=1),
        Row(id=3, first_name='Alex', second_name='Ioannides', floor=2),
        Row(id=4, first_name='Ken', second_name='Lai', floor=2),
        Row(id=5, first_name='Stu', second_name='White', floor=3),
        Row(id=6, first_name='Mark', second_name='Sweeting', floor=3),
        Row(id=7, first_name='Phil', second_name='Bird', floor=4),
        Row(id=8, first_name='Kim', second_name='Suter', floor=4)
    ]

    df = spark.createDataFrame(local_records)

    # write to Parquet file format
    (df
     .coalesce(1)
     .write
     .parquet('../tests/test_data/job_demo/employees_', mode='overwrite'))

    return None


# entry point for PySpark ETL application
if __name__ == '__main__':
    create_test_data()

