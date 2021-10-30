"""
job_realtime_3p_5_10.py
POP 实时预测 中销模型median5-10
app_sale_3p_middlesales_hour_model
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

def get_feature_array_with_threshold(row, config):
    """
    将sku真实销量转换为训练数据特征
    """
    import pandas as pd

    bu_id, wh_id, sku_id, hour, model_name = row[0]
    target_hours = 23 - hour
    columns = config['columns']
    up_threshold = config['up_threshold']
    low_threshold = config['low_threshold']

    # 构造预测维度原始df
    df = pd.DataFrame(row[1], columns=columns)
    df['gap_btime'] = pd.to_datetime(df['gap_btime'])
    df = df.set_index('gap_btime')

    try:
        # 生成特征df
        if df['arranged_cnt'].resample('D').sum().sort_index()[-8:-1].median() < low_threshold:
            return [None]
        if df['arranged_cnt'].resample('D').sum().sort_index()[-8:-1].median() > up_threshold:
            return [None]
        df_features = extract_sku_info(df, bu_id, wh_id, sku_id, target_hours=target_hours,avg_sale_limit = low_threshold)
    except:
        # 数据不全不参与训练
        return [None]

    return [None] if df_features is None or df_features.shape[0] == 0 else [(hour, model_name),
                                                                            df_features.values.tolist()]

def main():

    """Main script definition.
    :return: None
    """
    # start Spark application and get Spark session, logger and config
    spark, log, config = start_spark(app_name='realtime_3p_5_10 app_sale_3p_middlesales_hour_model')
    log.warn('realtime_3p_5_10 app_sale_3p_middlesales_hour_model is up-and-running')
    config['dt'] = pd.to_datetime(config['dt']) if 'dt' in config else datetime.datetime.now().date()
    config['label_start_date'] = config['dt'] + datetime.timedelta(days=-config['date_span'])
    config['dt'],config['label_start_date'] = config['dt'].strftime("%Y%m%d"),config['label_start_date'].strftime("%Y%m%d")
    print('realtime_3p_5_10 app_sale_3p_middlesales_hour_model is up-and-running time using is ',config['dt'],config['label_start_date'],config['low_threshold'],config['up_threshold'])
    # execute spark pipeline
    # 训练模型
    df_raw = extract_train_data(spark,config)

    # 增加hour字段并explode
    from pyspark.sql import functions as F
    from pyspark.sql.functions import explode

    df_raw = df_raw.withColumn('hour', F.array([F.lit(x) for x in range(12, 23)]))
    df_raw = df_raw.withColumn('hour', explode('hour'))

    # 要训练的模型列表
    model_name_list = ['xgb_realtime_3p_mid'] #, , 'gbdt_realtime_3p', 'dnn_realtime_3p'，'rf_realtime_3p', 'etr_realtime_3p']

    df_raw = df_raw.withColumn('model', F.array([F.lit(x) for x in model_name_list]))
    df_raw = df_raw.withColumn('model', explode('model'))

    print('df_raw.count() = ', df_raw.count())

    # 特征工程并训练
    config['columns'] = df_raw.columns
    rdd = df_raw.rdd.map(lambda row: ((row['bu_id'], row['wh_id'], row['sku_id'], row['hour'], row['model']), row))\
                            .groupByKey().map(lambda row: get_feature_array_with_threshold(row, config))\
                            .filter(lambda x: x[0] is not None).reduceByKey(combine_features).map(train_model)\
                            .filter(lambda x: x[0] is not None)
    
    # 写Hive结果
    write_data(spark, rdd, config)

    # log the success and terminate Spark application
    log.warn('realtime_3p_5_10 app_sale_3p_middlesales_hour_modelis finished')
    print('realtime_3p_5_10 app_sale_3p_middlesales_hour_modelis finished')
    spark.stop()

    return None


#############################################
# extract data 
#############################################

def extract_train_data(spark,config):

    sql = """
    select * from mart_caterb2b_forecast.app_sale_wh_bu_sku_min_input 
    where management_type = 3  
          and dt>='{label_start_date}' and  dt < '{dt}'
    order by dt, hour
    """.format(label_start_date=config['label_start_date'],dt=config['dt'])

    print('fetching sql is ',sql)

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
    # 没办法刷数，除非修正realtime utils 文件，待定吧
    # res_df.select('model_type', 'hour', 'body', 'dt', 'model_name').write.insertInto(config['hive_table'], overwrite=True)
    res_df.select("model_type", "hour", "body").write.mode('overwrite').orc(config['hive_path'].format(partition_dt=config['dt'],model_name=config['model_name']))


#############################################
# entry point for PySpark application
#############################################



if __name__ == '__main__':
    main()



