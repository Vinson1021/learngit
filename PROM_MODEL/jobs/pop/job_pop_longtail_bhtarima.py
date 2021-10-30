#!/bin/usr/env python3.6
# -*- coding: utf-8 -*-
# ---------------------------------
# ProjectName: data-forecast-spark
# Description: 3P 非实时长尾优化
# Author: lijingjie
# file:job_pop_longtail_bhtarima.py
# CreateTime: 2021-04-07 22:35
# ---------------------------------
# 设置随机性

import numpy as np
import random
import os
seed=44
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

import os, sys
import json
import datetime
import pandas as pd
import numpy as np
from itertools import chain
from pyspark.sql.types import *
from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import *
from functools import reduce
from pyspark.sql import DataFrame
from pyspark import SparkFiles

import numpy as np
from dependencies.platform.spark import start_spark
from dependencies.algorithm.pop.preprocess import extract_data, seqOp, combOp, save_to_hive
from dependencies.algorithm.pop.model import bht_arima

#### -----> 模型预测输出    支持多步预测
# 'bu_id,wh_id,sku_id,fc_dt,inter_fcst_rst'
#### -----> 最终输出结果表，需要调用函数 save_to_hive
schema = StructType([StructField("bu_id", LongType(), True), StructField("wh_id", LongType(), True),
                     StructField("sku_id", LongType(), True), StructField("fc_dt", StringType(), True),
                     StructField("fcst_rst", DoubleType(), True)])

#############################################
# main function
#############################################

def main():
    """Main script definition.
    :return: None
    """
    # start Spark application and get Spark session, logger and config
    spark, log, configs = start_spark(app_name='job_pop_longtail_bht_arima')
    # 日期参数设定
    configs['fc_dt'] = pd.to_datetime(configs['fc_dt']) if 'fc_dt' in configs else datetime.datetime.now().date()
    configs['today'] = pd.to_datetime(configs['fc_dt']) + datetime.timedelta(
        days=-1) if 'fc_dt' in configs else datetime.datetime.now().date() + datetime.timedelta(days=-1)

    configs['label_start_date'] = pd.to_datetime(
        configs['label_start_date']) if 'label_start_date' in configs else pd.to_datetime('20210220').date()
    configs['pred_dates'] = pd.date_range(start=pd.to_datetime(configs['fc_dt']),
                                          periods=configs['pred_len'])  # 跟预测步长match

    fitting_size = configs['fitting_size']
    lower_threshold = configs['lower_threshold']
    upper_threshold = configs['upper_threshold']

    # log that main spark job is starting
    log.warn('job_pop_longtail_bht_arima is up-and-running')
    print('job_pop_longtail_bht_arima is up-and-running', configs['fc_dt'], configs['today'], fitting_size)

    # STEP0 : ===============>获取数据
    raw_data, _ = extract_data(spark, configs)
#     raw_data = raw_data.filter(((F.col('wh_id') == 136))).persist()
    cnt_data_hist = raw_data.withColumn('unique_id', F.concat(F.col('bu_id'), F.lit('_'), F.col('wh_id'), F.lit('_'),
                                                              F.col('sku_id')))
    configs['cnt_data_columns'] = cnt_data_hist.schema.names

    overall_count_1 = cnt_data_hist.select("unique_id").distinct().count()

    # STEP1 : ===============>sku 动态划分区间
    log.info("Start longtail sku judgment logic...")
    cnt_data_hist.registerTempTable("cnt_data_hist_table")
    # get mean and saled days before labelling
    log.info("labelling groups")
    mean_saled_days = """
                        select dt,
                              unique_id,
                              arranged_cnt,
                              percentile(arranged_cnt, 0.5) over (partition by unique_id order by dt rows between 20 preceding and 0 preceding) as median_value,
                              avg(arranged_cnt) over (partition by unique_id order by dt rows between 20 preceding and 0 preceding) as mean_value,
                              count(arranged_cnt) over (partition by unique_id order by dt rows between 20 preceding and 0 preceding)  as count_value,
                              sum(arranged_cnt) over (partition by unique_id order by dt rows between 6 preceding and 0 preceding)  as sum_7_value
                         from cnt_data_hist_table
                        where dt >= '20210218'"""
    mean_saled_days_df = spark.sql(mean_saled_days)
    mean_saled_days_df.registerTempTable("mean_saled_days_df_table")

    cnt_data_with_mean_df = spark.sql("""
                        select a.*,b.mean_value,b.count_value from (select * 
                        from cnt_data_hist_table where arranged_cnt>0 or is_on_shelf=1) a
                        join (
                        select unique_id,mean_value,count_value
                               from mean_saled_days_df_table 
                        where dt=""" + configs['today'].strftime('%Y%m%d') + """ 
                        and mean_value>=""" + str(lower_threshold) + """ and mean_value<=""" + str(upper_threshold) + """ 
                        and count_value>=14 --contain 14 days which means that 13 if starts with 0
                        and sum_7_value>0 --recent 7 days has sale
                        )b 
                        on a.unique_id = b.unique_id
                        """)

    ### 得到每一个dt，row number的划分
    cnt_data_with_mean_df = cnt_data_with_mean_df.withColumn("row_number", F.row_number().over(
        Window.partitionBy("dt").orderBy("mean_value")))

    # predict day size
    row_number_max = \
        cnt_data_with_mean_df.filter(F.col('dt') == configs['today'].strftime('%Y%m%d')).agg(
            {"row_number": "max"}).collect()[0][0]

    dynamic_str = ' '.join(
        ["""when row_number <""" + str(i * fitting_size) + """ then 'group_""" + str(i) + """'""" for i in
         range(1, row_number_max // fitting_size + 2)])
    dynamic_str = """ case """ + dynamic_str + """
                      end as group_label"""
    cnt_data_with_mean_df.registerTempTable("cnt_data_with_mean_df_table")

    split_with_conf_size = """
    select dt,
          unique_id,
          mean_value,
          count_value,
          """ + dynamic_str + """
     from cnt_data_with_mean_df_table
    where dt = '""" + configs['today'].strftime('%Y%m%d') + """' --using today date for filtering 
    """
    cnt_data_with_mean_df_with_group = spark.sql(split_with_conf_size)#只有一天数据

    # STEP2 : ===============>区间内预测
    log.info("using bht_airma to predict")
    cnt_data_with_mean_df_with_group.registerTempTable("cnt_data_with_mean_df_with_group")

    checking_agg = spark.sql("""select group_label,max(mean_value),min(mean_value),count(mean_value) from cnt_data_with_mean_df_with_group group by group_label""").toPandas()
    print('job_pop_longtail_bht_arima,checking_agg',checking_agg)
    cnt_data_to_fit = spark.sql("""
                        select a.*,b.mean_value,b.count_value,b.group_label from (select * 
                        from cnt_data_with_mean_df_table ) a
                        join (select distinct unique_id,mean_value,count_value,group_label
                        from cnt_data_with_mean_df_with_group)b
                        on a.unique_id = b.unique_id
                        """)
    configs['bht_arima_columns'] = cnt_data_to_fit.columns
    configs['seed'] = seed
    print('job_pop_longtail_bht_arima,overall_count_1,row_number_max', overall_count_1, row_number_max,
          cnt_data_to_fit.count(), cnt_data_with_mean_df_with_group.count())
    # job_pop_longtail_bht_arima,overall_count_1,row_number_max 236285 40536 1294301 40536

    forecast_bht_arima = cnt_data_to_fit.rdd.map(
        lambda row: ((row['group_label']), list(row))).aggregateByKey(list(), seqOp, combOp).sortByKey(
        numPartitions=100, keyfunc=lambda x: x).flatMap(
        lambda item: bht_arima(item, configs)).filter(lambda x: x is not None).map(
        lambda x: (int(x[0]), int(x[1]), int(x[2]), str(x[3]), float(x[4]))).persist()

    print('forecast_bht_arima is ', forecast_bht_arima.take(10))
    forecast_bht_arima = spark.createDataFrame(forecast_bht_arima, schema)
    forecast_bht_arima_count = forecast_bht_arima.count()
    # STEP 3 final output
    ##### 最终输出 'bu_id','wh_id','sku_id','model','total_fc_cnt,daily_fc_cnt'
    print('job_pop_longtail_bht_arima,construct fianl output ', forecast_bht_arima.count(), forecast_bht_arima_count,forecast_bht_arima.take(1))
    save_to_hive(forecast_bht_arima, configs['fc_dt'].strftime('%Y%m%d'), model_id=configs['model_id'],
                 output_path=configs['output_path'], model_name=configs['model_name'])

    # log the success and terminate Spark application
    log.warn('job_pop_longtail_baseline baseline is finished')
    spark.stop()
    return None


#############################################
# entry point for PySpark application
#############################################
if __name__ == '__main__':
    main()

##################extra info#############

# CREATE TABLE `mart_caterb2b_forecast_test`.`app_sale_3p_fc_basic_day_output`(
# `bu_id` bigint COMMENT 'bu id',
# `wh_id` bigint COMMENT '仓库id',
# `sku_id` bigint COMMENT 'sku id',
# `model_id` int COMMENT '预测模型id',
# `today_fc_cnt` double COMMENT 'T0预测销量',
# `daily_fc_cnt` string COMMENT '每日预测销量')
# COMMENT '销量预测-3p预测结果表'
# PARTITIONED BY (`fc_dt` string COMMENT '日期分区', `model_name` string COMMENT '预测模型name')
# stored as orc;
