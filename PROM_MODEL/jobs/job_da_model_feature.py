"""
job_da_model_training.py
~~~~~~~~~~

This example script can be executed as follows,

    $SPARK_HOME/bin/spark-submit \
    --master local[5] \
    --py-files packages.zip \
    --files configs/dev_da_model_feature_config.json \
    jobs/job_da_model_feature.py

"""
# import sys
# import os
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)

from datetime import datetime

from pyspark import RDD
from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *

import dependencies.algorithm.da_model.data_augmentation as da
from dependencies.common.forecast_utils import get_date, date_format_compat
from dependencies.platform.spark import start_spark


#############################################
# main function
#############################################
def main():
    """
    Main script definition.

    Returns: None

    """
    # start Spark application and get Spark session, logger and config
    spark, log, config = start_spark(
        app_name='da_model_feature_engineering')

    # log that main spark job is starting
    log.warn('da_feature_engineering is up-and-running')

    # execute spark pipeline
    data_df = extract_data(spark, config)
    fcst_rst = transform_data(data_df, config)
    write_data(spark, fcst_rst, config)

    # log the success and terminate Spark application
    log.warn('da_feature_engineering is finished')
    spark.stop()
    return None


#############################################
# extract data
#############################################
def extract_data(spark: SparkSession, config: dict) -> DataFrame:
    """
    Load data from hive table.

    Args:
        spark: spark对象
        config: 配置参数

    Returns:
        data_df: 原始特征DataFrame
    """

    # 日期参数设定
    forecast_date = datetime.now().strftime("%Y%m%d")
    if 'forecast_date' in config:
        forecast_date = config['forecast_date']

    train_feat_end_date = get_date(forecast_date, -1, date_format_compat, date_format_compat)
    train_feat_start_date = get_date(train_feat_end_date, -730, date_format_compat, date_format_compat)

    train_sql = """
            select  
                    feat.is_train,
                    wh.bu_id,
                    feat.wh_id,
                    feat.sku_id,
                    feat.cat1_name,
                    feat.cat2_name,
                    cast(feat.cnt_band as string),
                    feat.brand_name,
                    feat.tax_rate,
                    feat.csu_origin_price,
                    feat.seq_num,
                    feat.seq_price,
                    feat.redu_num,
                    feat.pro_num,
                    feat.discount_price,
                    feat.day_abbr,
                    feat.is_work_day,
                    feat.is_weekend,
                    feat.is_holiday,
                    feat.festival_name,
                    feat.western_festival_name,
                    feat.weather,
                    feat.weather_b,
                    feat.avg_t,
                    feat.avg_t_b,
                    feat.new_byrs,
                    feat.old_byrs,
                    feat.poultry_byrs,
                    feat.vege_fruit_byrs,
                    feat.egg_aquatic_byrs,
                    feat.standard_byrs,
                    feat.vege_byrs,
                    feat.meet_byrs,
                    feat.frozen_meat_byrs,
                    feat.forzen_semi_byrs,
                    feat.wine_byrs,
                    feat.rice_byrs,
                    feat.egg_byrs,
                    feat.is_on_shelf,
                    feat.is_outstock,
                    feat.arranged_cnt,
                    feat.dt,
                   s.brand_id,
                   s.cat3_id
              from (
                    select *
                      from mart_caterb2b_forecast.app_caterb2b_forecast_sale_input_v3
                     where dt between '{start_date}' and '{end_date}'
                       and is_train = 1
                 union all select *
                      from mart_caterb2b_forecast.app_caterb2b_forecast_sale_input_v3
                     where dt >= {forecast_start_date}
                       and dt < {forecast_end_date}
                       and is_train = 0
                   ) feat
             inner join mart_caterb2b.dim_caterb2b_sku s
                on feat.sku_id = s.sku_id  
             inner join mart_caterb2b.dim_caterb2b_warehouse wh
                on feat.wh_id = wh.id
            where s.cat1_id in ({cat1_id_list})
        """.format(start_date=train_feat_start_date, end_date=train_feat_end_date, forecast_start_date=forecast_date,
                   forecast_end_date=get_date(forecast_date, 20, date_format_compat, date_format_compat),
                   cat1_id_list=config['cat1_ids'])
    normal_feat_cols = config['normal_feat_cols'].split(',')
    raw_feat_cols = spark.sql(train_sql).schema.names
    data_df = spark.sql(train_sql).rdd.persist()

    config.update({
        'forecast_date': forecast_date,
        'raw_feat_cols': raw_feat_cols,
        'normal_feat_cols': normal_feat_cols
    })

    return data_df


#############################################
# data processing
#############################################
def transform_data(data_df: DataFrame, config: dict) -> RDD:
    """
    transform data

    Args:
        data_df: 原始特征DataFrame
        config: 配置参数

    Returns:
        fcst_rst: 数据增强处理后的数据RDD

    """
    fcst_rst = data_df.map(
        lambda x: ((x[config['raw_feat_cols'].index('cat1_name')], x[config['raw_feat_cols'].index('wh_id')], x[config['raw_feat_cols'].index('bu_id')]), x)
    ).groupByKey().flatMap(
        lambda x: get_normal_feat(x, config['forecast_date'], config['raw_feat_cols'], config['normal_feat_cols'])
    ).filter(
        lambda x: x is not None
    ).map(lambda x: list(map(str, x))).persist()

    return fcst_rst


#############################################
# output data
#############################################
def write_data(spark: SparkSession, fcst_rst: RDD, config: dict):
    """
    save trained model to hive

    Args:
        spark: spark对象
        fcst_rst: 数据增强处理后的数据RDD
        config: 配置参数

    Returns: None

    """
    hive_path = config['result_path'].format(partition_dt=config['forecast_date'])
    schema = StructType([StructField(i, StringType(), True) for i in config['normal_feat_cols']])
    res_df = spark.createDataFrame(fcst_rst, schema).persist()
    res_df.select(config['normal_feat_cols']).repartition(100).write.mode('overwrite').orc(hive_path)
    return None


#############################################
# helper
#############################################
def get_normal_feat(line, forecast_date, raw_feat_cols, normal_feat_cols):
    tmp = da.train_map(line, forecast_date, raw_feat_cols, normal_feat_cols)
    if len(tmp) != 2:
        return [None]
    train_feat, predict_feat = tmp[0], tmp[1]
    flat_train_list = [sublist for sublist in train_feat]
    flat_predict_list = [item for sublist in predict_feat for item in sublist]
    flat_train_list += flat_predict_list
    return flat_train_list


#############################################
# entry point for PySpark application
#############################################
if __name__ == '__main__':
    main()
