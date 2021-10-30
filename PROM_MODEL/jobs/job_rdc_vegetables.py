"""
job_rdc_vegetables.py
RDC蔬菜品类销量预测
"""

from dependencies.platform.spark import start_spark

from pyspark import SparkFiles
from sklearn.externals import joblib


from dependencies.algorithm.rdc_vegetables.timeseries_forecast import holtwinters
from dependencies.algorithm.rdc_vegetables.feature_process import sku_predict_t0, get_train_model, extract_train_data, predict

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
    spark, log, config = start_spark(app_name='job_rdc_vegetables')

    # log that main spark job is starting
    log.warn('job_rdc_vegetables is up-and-running')

    # 当前系统日期
    if 'fc_dt' not in config:
        config['fc_dt'] = datetime.datetime.now().strftime("%Y%m%d")

    # execute spark pipeline

    # 针对所有蔬菜和鲜蛋一级品类
    config_cat2 = {
                        'fc_dt': config['fc_dt'], 'date_span': config['date_span'],
                        "wh_id_list": config['wh_id_list'], 
                        'cat2_id_list': "10021362"
                  }


    # 训练模型
    train_df = extract_train_data(spark)
    model_dict = get_train_model(train_df.rdd, train_df.columns)

    gbr_model = model_dict['gbr']
    lr_model = model_dict['lr']
    xgb_model = model_dict['xgb']

    spark.sparkContext.broadcast(gbr_model)
    spark.sparkContext.broadcast(lr_model)
    spark.sparkContext.broadcast(xgb_model)

    # gbr_model = joblib.load(SparkFiles.get('veg_gbr.dat'))
    # spark.sparkContext.broadcast(gbr_model)


    # 针对config_cat2中品类数据预测
    # 在所设定的整个品类上执行, 区分不同id便于配置不同品类下生效的输出模型
    # ses和moving_avg作为baseline
    data = extract_data(spark, config_cat2).cache()
    rdd1 = transform_data(data, config['fc_dt'], None, 'ses', 3002)
    rdd2 = transform_data(data, config['fc_dt'], None, 'moving_avg', 3003)
    rdd3 = transform_data(data, config['fc_dt'], xgb_model, 'xgb', 3004)

    # 只针对RDC仓和sku的算法
    data = extract_data(spark, config).cache()
    rdd4 = transform_data(data, config['fc_dt'], lr_model, 'lr', 3005)
    rdd5 = transform_data(data, config['fc_dt'], gbr_model, 'gbr', 3006)
    rdd6 = transform_data(data, config['fc_dt'], xgb_model, 'xgb', 3007)

    # 写Hive结果
    # write_data(spark, rdd2.union(rdd3).union(rdd4), config)
    write_data(spark, spark.sparkContext.union([rdd1, rdd2, rdd3, rdd4, rdd5, rdd6]), config)

    # log the success and terminate Spark application
    log.warn('job_demo is finished')
    spark.stop()

    return None


#############################################
# extract data 
#############################################
def extract_data(spark, config):
    """
    从Hive中读取历史数据
    :return: Spark DataFrame.
    """
    def get_cond_sql(variable_name, list_str):
        """
        依据sql条件变量个数, 来组合判断条件
        """
        return 'and %s in (%s)'%(variable_name, list_str) if len(list_str.split(','))>1 else 'and %s = %s'%(variable_name, list_str)

    # 生成查询条件相关信息
    import datetime
    start_dt = (datetime.datetime.strptime(config['fc_dt'], "%Y%m%d") - datetime.timedelta(days=config['date_span'])).strftime('%Y%m%d')

    # 生成各种查询条件
    wh_id_cond = get_cond_sql('wh_id', config['wh_id_list']) if 'wh_id_list' in config else ''
    cat1_id_cond = get_cond_sql('cat1_id', config['cat1_id_list']) if 'cat1_id_list' in config else ''
    cat2_id_cond = get_cond_sql('cat2_id', config['cat2_id_list']) if 'cat2_id_list' in config else ''
    sku_id_cond = get_cond_sql('sku_id', config['sku_id_list']) if 'sku_id_list' in config else ''


    sql = """
    select * from 
    (
      select t1.dt, t1.bu_id, t1.wh_id, t1.sku_id, t2.cat1_id, t2.cat2_id, t1.arranged_cnt, t1.on_shelf from
      mart_caterb2b_forecast.app_caterb2b_forecast_input_sales_dt_wh_sku t1
      left join mart_caterb2b.dim_caterb2b_sku t2
      on t1.sku_id = t2.sku_id
    )
    where on_shelf = 1
    and dt >= %s
    %s %s %s %s
    """%(start_dt, wh_id_cond, cat1_id_cond, cat2_id_cond, sku_id_cond)


    print(sql)

    # sql = """
    # select dt, bu_id, wh_id, sku_id, cat1_id, arranged_cnt, avg_t, pro_num, is_train from 
    # mart_caterb2b_forecast.app_caterb2b_forecast_sale_input_v3  
    # where dt >= '20200201'
    # and is_outstock = 0
    # and wh_id in (189, 101, 100, 263, 173)
    # and cat1_id = 10021361
    # and sku_id in (10480747, 10480754, 10498709, 10506947, 10506949, 10507928, 10507930, 10507931, 10511802, 
    # 10511804, 10512435, 10513370, 10514623, 10514624, 10516412, 10517450, 10518393)
    # order by dt
    # """
    # and is_outstock = 0

    return spark.sql(sql)


#############################################
# data processing 
#############################################
def transform_data(df, fc_dt, model, model_name, model_id):
    """Transform original dataset.

    :param df: Input DataFrame.
    :param config: Config params from json.
    :return: Transformed DataFrame.
    """

    # 训练数据列名
    columns = df.columns


    result_rdd = df.rdd.map(lambda row: ((row['bu_id'], row['wh_id'], row['sku_id']), list(row)))\
                                .groupByKey().map(lambda row: predict(row, columns, fc_dt, model, model_name, model_id))\
                                .filter(lambda x: x is not None)

    return result_rdd


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
        StructField("fc_dt", StringType(), True),
        StructField("bu_id", LongType(), True),
        StructField("wh_id", LongType(), True),
        StructField("sku_id", LongType(), True),
        StructField("source", IntegerType(), True),
        StructField("model", IntegerType(), True),
        StructField("total_amt_fc_wk", DoubleType(), True),
        StructField("daily_amt_fc_wk", StringType(), True)
    ])
    
    # 结果写入hive
    res_df = spark.createDataFrame(result_rdd, schema)
    
    res_df.select("bu_id", "wh_id", "sku_id", "source", "model", "total_amt_fc_wk", 
        "daily_amt_fc_wk").repartition(10).write.mode('overwrite').orc(config['hive_path'].format(partition_dt=config['fc_dt']))


#############################################
# entry point for PySpark application
#############################################
if __name__ == '__main__':
    main()

