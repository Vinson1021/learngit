#!/bin/usr/env python3.6
# -*- coding: utf-8 -*-
# ---------------------------------
# ProjectName: data-forecast-spark
# Description: 
# Author: songzhen07
# CreateTime: 2021-02-23 17:35
# ---------------------------------

from datetime import datetime
from functools import reduce

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window

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
        app_name='application_sale_forecast_degradation')

    # log that main spark job is starting
    log.warn('application_sale_forecast_degradation is up-and-running')

    # execute spark pipeline
    cnt_df, output_df, config = extract_data(spark, config)
    if len(output_df.head(1)) > 0:
        rst_df = transform_data(cnt_df, output_df, config)
        write_data(rst_df, config)
    else:
        log.warn("all prediction tables are empty!")

    # log the success and terminate Spark application
    log.warn('application_sale_forecast_degradation is finished')
    spark.stop()
    return None


#############################################
# extract data
#############################################
def extract_data(spark: SparkSession, config: dict) -> (DataFrame, DataFrame, dict):
    """
    get original data
    Args:
        spark: SparkSession
        config: config dict

    Returns:
        cnt_df: arranged cnt DataFrame
        output_df: prediction DataFrame
    """
    fc_dt = datetime.now().strftime("%Y%m%d")
    if 'fc_dt' in config:
        fc_dt = config['fc_dt']

    # 刷数设置
    # end_datetime = pd.to_datetime(fc_dt, format="%Y%m%d")-pd.Timedelta(days=1)

    # 历史销量真值数据
    df_cnt_list = []
    for table_name in config['feat_tables']:
        cnt_df = spark.sql('''
            select 
                sku_id, 
                bu_id, 
                wh_id,
                date,
                {model_id} as model,
                arranged_cnt
            from {feat_table}
            where is_train = 1 
              and dt = {fc_dt}
        '''.format(model_id=config['feat_tables'][table_name], feat_table=table_name, fc_dt=fc_dt))
        df_cnt_list.append(cnt_df)
    cnt_df = reduce(lambda a, b: a.union(b), df_cnt_list)

    # 刷数设置，正式上线该值赋值fc_dt
    # delta_fc_dt = end_datetime.strftime("%Y%m%d")
    delta_fc_dt = fc_dt

    # 原始预测结果
    df_rst_list = []
    for table_name in config['output_tables']:
        output_df = spark.sql('''
            select 
                sku_id,
                wh_id,
                bu_id,
                date,
                prediction
            from {output_table}
            where dt = {delta_fc_dt}
        '''.format(output_table=table_name, delta_fc_dt=delta_fc_dt))
        df_rst_list.append(output_df)
    output_df = reduce(lambda a, b: a.union(b), df_rst_list)

    config.update({"fc_dt": fc_dt})
    return cnt_df, output_df, config


#############################################
# data processing
#############################################
def transform_data(cnt_df: DataFrame, output_df: DataFrame, config: dict) -> DataFrame:
    """
    prediction degradation processing
    Args:
        cnt_df: arranged cnt DataFrame
        output_df: prediction DataFrame
        config: config dict

    Returns:
        rst_df: result DataFrame
    """
    window = Window.partitionBy(cnt_df['sku_id'], cnt_df['bu_id'], cnt_df['wh_id'], cnt_df['model']) \
        .orderBy(cnt_df['date'].desc())
    cnt_mean_df = cnt_df.select('*', F.rank().over(window).alias('rank')) \
        .filter(F.col('rank') <= config['degradation_interval']) \
        .groupby(['sku_id', 'wh_id', 'bu_id', 'model']) \
        .agg(F.mean('arranged_cnt').alias('cnt_avg'))

    rst_df = output_df.join(cnt_mean_df, ['sku_id', 'wh_id', 'bu_id'], 'inner')
    # 降级逻辑
    rst_df = rst_df.withColumn("rst", F.when(F.col("prediction") > 3.0 * F.col("cnt_avg"), 3.0 * F.col("cnt_avg"))
                               .when(F.col("prediction") < F.col("cnt_avg") / 3.0, F.col("cnt_avg") / 3.0)
                               .otherwise(F.col("prediction"))).withColumn("fc_dt", F.lit(config['fc_dt']))
    # 预测数据融合
    sum_win = Window.partitionBy('sku_id', 'wh_id', 'bu_id', 'model')
    agg_df = rst_df.withColumn('total_amt_fc_wk', F.sum('rst').over(sum_win)) \
        .groupby('sku_id', 'wh_id', 'bu_id', 'model', 'total_amt_fc_wk') \
        .pivot('date').agg(F.first('rst').alias('pred'))
    rst_df = agg_df.withColumn('daily_amt_fc_wk', F.regexp_replace(
        F.to_json(F.struct([F.when(F.lit(x).like('20%'), agg_df[x]).alias(x) for x in agg_df.columns])), 'pred', ''))
    return rst_df


#############################################
# output data
#############################################
def write_data(rst_df: DataFrame, config: dict) -> None:
    hive_path = config['rst_hive_path'].format(partition_dt=config['fc_dt'])
    rst_df.select(F.col('bu_id'), F.col('wh_id'), F.col('sku_id'), F.lit(config['source']).alias('source'),
                  F.col('model'), F.col('total_amt_fc_wk'), F.col('daily_amt_fc_wk')
                  ).repartition(100).write.mode('overwrite').orc(hive_path)
    return None


#############################################
# entry point for PySpark application
#############################################
if __name__ == '__main__':
    main()