"""
job_rdc_vegetables.py
RDC蔬菜品类销量预测
"""

from dependencies.platform.spark import start_spark

from pyspark import SparkFiles

from dependencies.algorithm.realtime_3p.realtime_utils import *

import datetime
import pandas as pd
import json


#############################################
# main function
#############################################

def main():

    """Main script definition.

    :return: None
    """
    # start Spark application and get Spark session, logger and config
    spark, log, config = start_spark(app_name='realtime_3p')

    # log that main spark job is starting
    log.warn('realtime_3p is up-and-running')

    # execute spark pipeline

    # 训练模型
    df_raw = extract_train_data(spark)

    # 增加hour字段并explode
    from pyspark.sql import functions as F
    from pyspark.sql.functions import explode

    df_raw = df_raw.withColumn('hour', F.array([F.lit(x) for x in range(12, 23)]))
    df_raw = df_raw.withColumn('hour', explode('hour'))

    # 要训练的模型列表
    model_name_list = ['xgb_realtime_3p_high', 'gbdt_realtime_3p_high', 'dnn_realtime_3p_high'] #, 'rf_realtime_3p', 'etr_realtime_3p']

    df_raw = df_raw.withColumn('model', F.array([F.lit(x) for x in model_name_list]))
    df_raw = df_raw.withColumn('model', explode('model'))

    # print('df_raw.count() = ', df_raw.count())

    config['dt'] = datetime.datetime.now().strftime("%Y%m%d")

    # 特征工程并训练
    columns = df_raw.columns
    rdd = df_raw.rdd.map(lambda row: ((row['bu_id'], row['wh_id'], row['sku_id'], row['hour'], row['model']), row))\
                            .groupByKey().map(lambda row: get_feature_array(row, columns))\
                            .filter(lambda x: x[0] is not None).reduceByKey(combine_features).map(train_model)\
                            .filter(lambda x: x[0] is not None)
    
    # 写Hive结果
    write_data(spark, rdd, config)

    # log the success and terminate Spark application
    log.warn('job_demo is finished')
    spark.stop()

    return None


#############################################
# extract data 
#############################################

def extract_train_data(spark):

    sql = """
    select * from mart_caterb2b_forecast.app_sale_wh_bu_sku_min_input 
    where management_type = 3 
    order by dt, hour
    """

    return spark.sql(sql)


#############################################
# data processing 
#############################################



#############################################
# output data
#############################################
def write_data(spark, result_rdd, config):
    """Collect data locally and write to CSV.

    :param df: DataFrame to print.
    :return: None
    """

    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, DoubleType
    schema = StructType([
        StructField('dt', StringType(), True),
        StructField('model_name', StringType(), True),
        StructField('model_type', StringType(), True),
        StructField('hour', IntegerType(), True),
        StructField('body', StringType(), True)
    ])

    # 结果写入hive
    res_df = spark.createDataFrame(result_rdd, schema)

    # print(res_df.show(5))
    
    # res_df.select("model_name", "model_type", "hour", "body").write.mode('overwrite').orc(config['hive_path'].format(partition_dt=config['dt']))
    spark.sql("set hive.exec.dynamic.partition=true")
    spark.sql("set hive.exec.dynamic.partition.mode=nostrick")
    res_df.select('model_type', 'hour', 'body', 'dt','model_name').write.insertInto(config['hive_table'], overwrite=True)


#############################################
# entry point for PySpark application
#############################################


if __name__ == '__main__':
    main()



