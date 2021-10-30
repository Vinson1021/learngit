#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------
# File  : job_pop_day_xgb.py
# Author: liushichang
# Date  : 2021/5/29
# Desc  : 3P 离线预测（XGBoost）
# Contact : liushichang@meituan.com
# ----------------------------------

import gc
from datetime import datetime

import pandas as pd
import pyspark.sql.functions as F
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import Row

from dependencies.algorithm.pop.sale_pop_xgb_t0 import SalePopXgbForecast
from dependencies.common import forecast_utils as fu
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
        app_name='data-forecast-3p-day-baseline')

    # log that main spark job is starting
    log.warn('job is up-and-running')

    # execute spark pipeline
    raw_data_df, config = extract_data(spark, config)
    res_df, config = transform_data(spark, raw_data_df, config)
    write_data(spark, res_df, config)

    # log the success and terminate Spark application
    log.warn('job is finished')
    spark.stop()
    return None


#############################################
# extract data
#############################################
def extract_data(spark: SparkSession, config: dict) -> (RDD, dict):
    """
    extract data from hive

    Args:
        spark: sparkSession
        config: config dict

    Returns:
        raw_data_rdd: data rdd
        config: updated config dict
    """

    # 日期参数设定
    current_date = datetime.now().strftime("%Y%m%d")
    fc_date_delta = 0
    train_duration = -540
    test_duration = 0
    if 'current_date' in config:
        current_date = config['current_date']
    if 'fc_date_delta' in config:
        fc_date_delta = int(config['fc_date_delta'])
    if 'train_duration' in config:
        train_duration = int(config['train_duration'])
    if 'test_duration' in config:
        test_duration = int(config['test_duration'])
    current_date = fu.get_date(current_date, fc_date_delta, '%Y%m%d', '%Y%m%d')
    train_start_date = fu.get_date(current_date, train_duration, '%Y%m%d', '%Y%m%d')
    train_end_date = fu.get_date(current_date, -1, '%Y%m%d', '%Y%m%d')
    test_end_date = fu.get_date(current_date, test_duration, '%Y%m%d', '%Y%m%d')

    extract_sql = """
        select {feature_cols}
         from mart_caterb2b_forecast.app_caterb2b_forecast_sale_3p_input 
         where 
            is_train=1 
            and dt between '{train_start_date}' and '{train_end_date}'
        union all 
        select {feature_cols}
         from mart_caterb2b_forecast.app_caterb2b_forecast_sale_3p_input 
         where 
            is_train=0 
            and dt between '{test_start_date}' and '{test_end_date}'    
    """.format(feature_cols=config['raw_feature_cols'],
               train_start_date=train_start_date,
               train_end_date=train_end_date,
               test_start_date=current_date,
               test_end_date=test_end_date
               )
    # 得到原始数据集
    feats_df = spark.sql(extract_sql)
    schema_cols = feats_df.schema.names

    new_configs = {
        'schema_cols': schema_cols,
        'current_date': current_date
    }
    config.update(new_configs)

    return feats_df, config


def seqOp(x, y):
    """seqFunc"""
    x.append(y)
    return x


def combOp(x, y):
    """combFunc"""
    return x + y


#############################################
# data processing
#############################################
def transform_data(spark: SparkSession, raw_data_df: DataFrame, config: dict) -> (DataFrame, dict):
    """
       transform data

       Args:
           spark: Spark session
           raw_data_df: raw data df
           config: config dict

       Returns:
           feats_rdd: features rdd
           config: updated config dict
       """
    config['key_cols'] = ['wh_id', 'bu_id', 'sku_id', 'dt']
    # 特征排序
    left_features = list(set(config['features']) - set(config['key_cols']))
    ordered_cols = ['cnt(t-%d)' % i for i in range(14, 0, -1)] + left_features + ['cnt(t-0)']
    # 列限制
    config['ordered_cols'] = ordered_cols
    config['train_data_cols'] = ['wh_id', 'bu_id', 'sku_id']
    config['longtail_data_cols'] = ['wh_id', 'bu_id', 'sku_id']
    config['raw_data_cols'] = ['dt', 'wh_id', 'bu_id', 'sku_id', 'cat1_name', 'is_train', 'arranged_cnt']
    config['pred_cols'] = ['bu_id', 'sku_id', 'wh_id', 'dt', 'prediction']

    sale_pop_baseline_forecast = SalePopXgbForecast(config["current_date"])

    preprocessed_rdd = raw_data_df.rdd.map(lambda row: ((row['wh_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=100, keyfunc=lambda x: x) \
        .map(lambda item: (item[0], sale_pop_baseline_forecast.preprocess_data(item[1], config)))\
        .filter(lambda t: t[1] is not None).persist()

    longtail_rdd = preprocessed_rdd.sortByKey(numPartitions=100, keyfunc=lambda x: x)\
        .map(lambda t: (t[0], sale_pop_baseline_forecast.process_longtail_data(t[1]))).persist()

    features_rdd = preprocessed_rdd.leftOuterJoin(longtail_rdd)\
        .sortByKey(numPartitions=100, keyfunc=lambda x: x)\
        .flatMap(lambda t: sale_pop_baseline_forecast.generate_features(t[1][0], t[1][1])).persist()

    train_df = preprocessed_rdd.values().flatMap(lambda x: x).toDF().filter(F.col('is_train') == 1).persist()
    longtail_df = longtail_rdd.values().flatMap(lambda x: x).toDF().persist()

    fu.log_print("XGB 阶段开始...")
    xgb_forecast = features_rdd.map(lambda x: ((0 if x['cnt_band'] == 0 else 1), [x])) \
        .reduceByKey(lambda a, b: a+b) \
        .flatMap(lambda item: train_n_predict(item[1], config)).toDF()

    fu.log_print("XGB 阶段结束！LT 阶段开始...")
    lt_forecast = preprocess_lt_date(spark, raw_data_df, train_df, longtail_df, xgb_forecast, config)
    fu.log_print("LT 阶段结束!")
    final_rst = xgb_forecast.select(*config['pred_cols']).union(
        lt_forecast.select(*config['pred_cols'])).dropDuplicates(subset=['wh_id', 'bu_id', 'sku_id'])
    fu.log_print("融合阶段结束!")

    return final_rst, config


#############################################
# output data
#############################################
def write_data(spark: SparkSession, res_df: DataFrame, config: dict) -> None:
    """
    save prediction result to hive

    Args:
        spark: sparkSession
        res_df: result
        config: config dict

    Returns:
        None
    """
    sum_window = Window.partitionBy('bu_id', 'wh_id', 'sku_id')
    agg_df = res_df.withColumn('today_fc_cnt', F.sum('prediction').over(sum_window)) \
        .groupby('bu_id', 'wh_id', 'sku_id', 'today_fc_cnt') \
        .pivot('dt') \
        .agg(F.first('prediction').alias('fcDt'))
    result_df = agg_df.withColumn('daily_fc_cnt',
                                  F.regexp_replace(
                                      F.to_json(F.struct(
                                          [F.when(F.lit(x).like('20%'), agg_df[x]).alias(x) for x in agg_df.columns])),
                                      'fcDt', '')) \
        .select(
        F.col('bu_id'),
        F.col('wh_id'),
        F.col('sku_id'),
        F.lit(config['model_id']).alias('model_id'),
        F.col('today_fc_cnt'),
        F.col('daily_fc_cnt'),
        F.lit(config['current_date']).alias('fc_dt'),
        F.lit(config['model_name']).alias('model_name'))

    # save to hive
    fu.log_print("写入 Hive 阶段!")
    spark.sql("set hive.exec.dynamic.partition=true")
    spark.sql("set hive.exec.dynamic.partition.mode=nostrick")
    result_df.repartition(10).write.insertInto(config['output_table'], overwrite=True)

    return None


#############################################
# helper
#############################################
def train_n_predict(features_rdd: RDD, config: dict) -> list:
    """
    XGB 训练&预测
    Args:
        features_rdd: 特征数据 RDD
        config: 配置项字典

    Returns:
        训练好的模型
    """
    fu.log_print("XGB 训练准备...")
    total_data = pd.DataFrame(features_rdd)

    # 特征编码
    # 将事业部和仓合并后统一编码
    map_fields = ['cat1_name', 'cat2_name', 'day_abbr', 'festival_name', 'brand_name', 'day_weather', 'night_weather']
    total_data[map_fields] = total_data[map_fields].astype('category')
    total_data[map_fields] = total_data[map_fields].apply(lambda x: x.cat.codes)
    df_train = total_data[total_data.is_train == 1]
    df_test = total_data[total_data.is_train == 0]
    df_train.drop('is_train', inplace=True, axis=1)
    df_test.drop('is_train', inplace=True, axis=1)

    # 删除多余数据
    del total_data
    gc.collect()

    fu.log_print("XGB 训练开始...")
    # 模型训练
    model = SalePopXgbForecast.train_xgb_model(df_train, config)

    fu.log_print("XGB 预测开始...")
    # 模型预测
    lr_forecast = SalePopXgbForecast.xgb_predict(model, df_test, config)
    fu.log_print("XGB 预测完成！")
    return lr_forecast.to_dict('records')


def preprocess_lt_date(
        spark: SparkSession,
        raw_df: DataFrame,
        train_data: DataFrame,
        longtail_data: DataFrame,
        xgb_forecast: DataFrame,
        config: dict) -> DataFrame:
    """
    处理长尾数据
    Args:
        spark: Spark Session
        raw_df: 原始数据 DataFrame
        train_data: 训练集 DataFrame
        longtail_data: 长尾数据 DataFrame
        xgb_forecast: xgb 预测结果
        config: 配置项字典

    Returns:

    """
    # LR未预测SKU
    tr_skus = train_data.dropDuplicates(subset=['wh_id', 'bu_id', 'sku_id']).select('wh_id', 'bu_id', 'sku_id')
    xgb_skus = xgb_forecast.dropDuplicates(subset=['wh_id', 'bu_id', 'sku_id']).select('wh_id', 'bu_id', 'sku_id')
    xgb_skus = xgb_skus.withColumn('lr', F.lit(1))

    merged = tr_skus.join(xgb_skus, how='left', on=['wh_id', 'bu_id', 'sku_id'])
    leave_skus = merged.filter(F.col('lr').isNull()).drop(F.col("lr"))

    # 和长尾合并
    lt_skus = leave_skus \
        .join(longtail_data.select('wh_id', 'bu_id', 'sku_id'), how='outer', on=['wh_id', 'bu_id', 'sku_id'])

    cols = ['dt', 'wh_id', 'bu_id', 'sku_id', 'cat1_name', 'arranged_cnt']
    cnt_longtail = raw_df.filter(F.col('is_train') == 1).join(lt_skus, how='inner', on=lt_skus.schema.names)\
        .select(*cols).orderBy('dt')

    # 长尾数据处理
    config['cnt_longtail_cols'] = cnt_longtail.schema.names
    cnt_avg_rdd = cnt_longtail.rdd.map(lambda row: ((row['wh_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=100, keyfunc=lambda x: x) \
        .flatMap(lambda item: SalePopXgbForecast.get_avg_result(item, config)).persist()
    cnt_avg_df = spark.createDataFrame(cnt_avg_rdd)
    lt_data = cnt_longtail.join(cnt_avg_df, how='inner', on=['bu_id', 'wh_id', 'sku_id', 'cat1_name'])

    # 长尾预测
    config['lt_data_cols'] = lt_data.schema.names
    lt_forecast = lt_data.rdd.map(lambda row: ((row['wh_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=100, keyfunc=lambda x: x) \
        .flatMap(lambda item: SalePopXgbForecast.gen_lt_prediction(item, config)).persist()
    return lt_forecast.map(lambda x: Row(**x)).toDF()


#############################################
# entry point for PySpark application
#############################################
if __name__ == '__main__':
    main()
