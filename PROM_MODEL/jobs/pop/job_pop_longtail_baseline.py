#!/bin/usr/env python3.6
# -*- coding: utf-8 -*-
# ---------------------------------
# ProjectName: data-forecast-spark
# Description: 3P 非实时长尾预测
# Author: lijingjie
# file:job_pop_longtail_baseline.py
# CreateTime: 2021-03-27 17:35
# ---------------------------------
import os, sys
import json
import datetime
import pandas as pd
import numpy as np
from itertools import chain
from pyspark.sql.types import *
from pyspark.sql import Window
import pyspark.sql.functions as F
from functools import reduce
from pyspark.sql import DataFrame
from pyspark import SparkFiles

from dependencies.platform.spark import start_spark
from dependencies.algorithm.pop.preprocess import extract_data,seqOp,combOp,save_to_hive
from dependencies.algorithm.pop.model import ses,ses_new

#### -----> 模型预测输出    支持多步预测
# 'bu_id,wh_id,sku_id,fc_dt,inter_fcst_rst'
#### -----> 最终输出结果表，需要调用函数 save_to_hive
schema = StructType([StructField("bu_id", LongType(), True), StructField("wh_id", LongType(), True),
                         StructField("sku_id", LongType(), True), StructField("fc_dt", StringType(), True),
                         StructField("fcst_rst", DoubleType(), True)])

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

#############################################
# main function
#############################################

def main():
    """Main script definition.
    :return: None
    """
    # start Spark application and get Spark session, logger and config
    spark, log, configs = start_spark(app_name='job_pop_longtail_baseline')
    # 日期参数设定
    configs['fc_dt'] = pd.to_datetime(configs['fc_dt']) if 'fc_dt' in configs else datetime.datetime.now().date()
    configs['label_start_date'] = pd.to_datetime(configs['label_start_date']) if 'label_start_date' in configs else pd.to_datetime('20210220').date()
    configs['pred_dates'] = pd.date_range(start=pd.to_datetime(configs['fc_dt']),periods=configs['pred_len'])# 跟预测步长match
    # log that main spark job is starting
    log.warn('job_pop_longtail_baseline is up-and-running')
    print('job_pop_longtail_baseline is up-and-running')

    # STEP0 : ===============>获取数据
    raw_data,_ = extract_data(spark, configs)

    cnt_data_hist = raw_data.withColumn('unique_id',F.concat(F.col('bu_id'), F.lit('_'), F.col('wh_id'), F.lit('_'), F.col('sku_id')))
    configs['cnt_data_columns'] = cnt_data_hist.schema.names

    overall_count_1 = cnt_data_hist.select("unique_id").distinct().count()

    # STEP1 : ===============>sku 基线长尾预测
    forecast_baseline = cnt_data_hist.rdd.map(
        lambda row: ((row['unique_id']), list(row))).aggregateByKey(list(), seqOp, combOp).sortByKey(numPartitions=1000,keyfunc=lambda x: x).flatMap(
        lambda item: ses_new(item, configs)).filter(lambda x: x is not None).map(
        lambda x: (int(x[0]), int(x[1]), int(x[2]), str(x[3]), float(x[4]))).persist()
    forecast_baseline_df = spark.createDataFrame(forecast_baseline, schema)
    forecast_baseline_df_count = forecast_baseline_df.count()
    print('job_pop_longtail_baseline,overall_count_1, forecast_baseline_df_count',overall_count_1, forecast_baseline_df_count)

    # STEP 3 final output
    ##### 最终输出 'bu_id','wh_id','sku_id','model','total_fc_cnt,daily_fc_cnt'
    print('job_pop_longtail_baseline,construct fianl output ',forecast_baseline_df.count())
    # save_to_hive(forecast_baseline_df,configs['fc_dt'].strftime('%Y%m%d'),model_id=1,model_name='ses')
    save_to_hive(forecast_baseline_df,configs['fc_dt'].strftime('%Y%m%d'),model_id=configs['model_id'],output_path=configs['output_path'],model_name=configs['model_name'])

    # log the success and terminate Spark application
    log.warn('job_pop_longtail_baseline baseline is finished')
    spark.stop()
    return None

#############################################
# entry point for PySpark application
#############################################
if __name__ == '__main__':
    main()

