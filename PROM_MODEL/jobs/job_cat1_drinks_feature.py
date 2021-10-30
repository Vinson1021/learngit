#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------
# File  : job_cat1_drinks_feature.py
# Author: liushichang
# Date  : 2021/3/23
# Desc  : 饮品品类模型特征工程
# Contact : liushichang@meituan.com
# ----------------------------------
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel

from dependencies.algorithm.cat_model.cat1_common_data_process import SaleDataPreprocess
from dependencies.algorithm.cat_model.cat1_common_derived_features import DerivedFeature
from dependencies.common.ops import seqOp, combOp
from dependencies.platform.spark import start_spark

random.seed(42)
np.random.seed(42)


#############################################
# main function
#############################################
def main():
    """Main script definition.

    :return: None
    """
    # start Spark application and get Spark session, logger and config
    spark, log, configs = start_spark(
        app_name='sale_forecast_cat1_drinks_feature_engineering')

    # log that main spark job is starting
    log.warn('cat1_drinks_feature_engineering is up-and-running')

    # execute spark pipeline
    config, data = extract_data(spark, configs)
    final_df = transform_data(spark, log, data, config)
    write_data(final_df, config)

    # log the success and terminate Spark application
    log.warn('cat1_drinks_feature_engineering is finished')
    spark.stop()
    return None


#############################################
# extract data
#############################################
def extract_data(spark, configs):
    """Load data from hive table.

    :param:
        spark: spark对象
        configs：配置参数

    :return:
        configs, df
    """
    # 日期参数设定
    current_date = datetime.now().strftime("%Y%m%d")
    if 'forecast_date' in configs:
        current_date = configs['forecast_date']

    # 预测日期
    forecast_date = (pd.to_datetime(current_date) + timedelta(configs['fc_date_delta'])).strftime('%Y%m%d')
    # 训练开始/结束日期
    train_start_date = (pd.to_datetime(forecast_date) + timedelta(configs['train_duration'])).strftime('%Y%m%d')
    train_end_date = (pd.to_datetime(forecast_date) - timedelta(1)).strftime('%Y%m%d')
    # 测试结束日期
    forecast_end_date = (pd.to_datetime(forecast_date) + timedelta(configs['test_duration'])).strftime('%Y%m%d')

    inquire_sql = '''
        select v.is_train,v.bu_id,v.wh_id,v.sku_id,v.cat1_name,v.cat2_name,v.cnt_band,v.brand_name,v.tax_rate,v.csu_origin_price,v.w_price,v.seq_num,v.seq_price,v.redu_num,v.csu_redu_num,v.cir_redu_num,v.pro_num,v.discount_price,v.day_abbr,v.is_work_day,v.is_weekend,v.is_holiday,v.festival_name,v.western_festival_name,v.weather,v.weather_b,v.avg_t,v.avg_t_b,v.new_byrs,v.old_byrs,v.poultry_byrs,v.vege_fruit_byrs,v.egg_aquatic_byrs,v.standard_byrs,v.vege_byrs,v.meet_byrs,v.frozen_meat_byrs,v.forzen_semi_byrs,v.wine_byrs,v.rice_byrs,v.egg_byrs,v.is_on_shelf,v.is_outstock,v.arranged_cnt,v.dt
          from mart_caterb2b_forecast.app_caterb2b_forecast_sale_input_v3 v
          join (
            select distinct wh_id, bu_id
              from mart_caterb2b.dim_warehouse_goods_owner_warehouse_relation 
             where goods_owner_type = 1 and dt = {train_end_date})  w
           on v.wh_id = w.wh_id and v.bu_id = w.bu_id
         where v.dt between {train_start_date} and {train_end_date}
           and v.is_train = 1
           and v.cat1_id = {cat1_id}
         union all 
        select v.is_train,v.bu_id,v.wh_id,v.sku_id,v.cat1_name,v.cat2_name,v.cnt_band,v.brand_name,v.tax_rate,v.csu_origin_price,v.w_price,v.seq_num,v.seq_price,v.redu_num,v.csu_redu_num,v.cir_redu_num,v.pro_num,v.discount_price,v.day_abbr,v.is_work_day,v.is_weekend,v.is_holiday,v.festival_name,v.western_festival_name,v.weather,v.weather_b,v.avg_t,v.avg_t_b,v.new_byrs,v.old_byrs,v.poultry_byrs,v.vege_fruit_byrs,v.egg_aquatic_byrs,v.standard_byrs,v.vege_byrs,v.meet_byrs,v.frozen_meat_byrs,v.forzen_semi_byrs,v.wine_byrs,v.rice_byrs,v.egg_byrs,v.is_on_shelf,v.is_outstock,v.arranged_cnt,v.dt
          from mart_caterb2b_forecast.app_caterb2b_forecast_sale_input_v3 v
          join (
            select distinct wh_id, bu_id
              from mart_caterb2b.dim_warehouse_goods_owner_warehouse_relation 
             where goods_owner_type = 1 and dt = {train_end_date})  w
           on v.wh_id = w.wh_id and v.bu_id = w.bu_id
         where v.dt between {forecast_date} and {forecast_end_date}
           and v.is_train = 0
           and v.cat1_id = {cat1_id}
    '''.format(train_start_date=train_start_date, train_end_date=train_end_date, cat1_id=configs['cat1_id'],
               forecast_date=forecast_date, forecast_end_date=forecast_end_date)
    # 得到原始数据集
    raw_data = spark.sql(inquire_sql).withColumnRenamed('dt', 'date')

    bd_col_names = raw_data.schema.names

    new_conf = {
        "current_date": current_date,
        "forecast_date": forecast_date,
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "forecast_end_date": forecast_end_date,
        "bd_col_names": bd_col_names
    }
    configs.update(new_conf)

    return configs, raw_data


#############################################
# data processing
#############################################
def transform_data(spark, log, raw_df, config):
    """
    生产特征数据

    :param:
        spark: spark对象
        log：日志打印对象
        raw_df: 输入spark df数据
        configs： 配置参数字典对象

    :return：
        rst_df: 训练+测试特征数据
    """

    # 数据预处理
    log.info("Start to preprocess...")
    buwh_rdd = raw_df.rdd.map(lambda row: ((row['bu_id'], row['wh_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=100, keyfunc=lambda x: x) \
        .flatMap(lambda item: data_preprocess(item, config))
    buwh_df = spark.createDataFrame(buwh_rdd).persist(StorageLevel.MEMORY_AND_DISK)

    log.info("Update configs by buwh schema...")
    buwh_col_names = buwh_df.schema.names
    config.update({"buwh_col_names": buwh_col_names})

    # 滑动过滤异常值
    log.info("Start filter outliers by windows operation...")
    cleaned_rdd = buwh_df.rdd.map(lambda row: ((row['bu_id'], row['wh_id'], row['sku_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=100, keyfunc=lambda x: x) \
        .flatMap(lambda item: drop_Outliers(item, config))
    cleaned_df = spark.createDataFrame(cleaned_rdd).persist(StorageLevel.MEMORY_AND_DISK)

    # 衍生特征制造
    feature_rdd = cleaned_df.rdd.map(lambda row: ((row['bu_id'], row['wh_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=100, keyfunc=lambda x: x) \
        .flatMap(lambda item: features_create(item, config))
    feature_df = spark.createDataFrame(feature_rdd).persist(StorageLevel.MEMORY_AND_DISK)

    # 特征编码
    map_list = config['label_cols']
    renamed_list = [col + '_index' for col in map_list]
    name_mapper = dict(zip(renamed_list, map_list))
    index_cols = list(map(lambda x: (F.dense_rank().over(Window.orderBy(x)) - 1).cast('double').alias(x+'_index'), map_list))
    final_df = feature_df.select(*(feature_df.columns+index_cols))

    # 删除/重命名特征列
    final_df = final_df.drop(*map_list)
    final_df = final_df.select([F.col(c).alias(name_mapper.get(c, c)) for c in final_df.columns])

    return final_df


#############################################
# output data
#############################################
def write_data(df, config):
    """
    write data to hive
    Args:
        df: spark dataframe
        config: configuration dict

    Returns:
        None
    """
    df.select(config['output_cols']).repartition(10).write.mode('overwrite').orc(
        config['output_path'].format(str(config['forecast_date']))
    )
    return None


#############################################
# helper
#############################################
def data_preprocess(raw_rdd, config):
    """
    数据预处理
    Args:
        raw_rdd: 原始数据rdd
        config: 配置dict

    Returns:
        预测后的数据
    """
    preprocess_obj = SaleDataPreprocess(config['cat1_id'], config['forecast_date'])

    # 转换为pandas DataFrame
    df_raw = pd.DataFrame(raw_rdd[1], columns=config['bd_col_names']).replace('', np.nan)
    df_raw['dt'] = pd.to_datetime(df_raw['date'])
    bu_id = df_raw.bu_id.unique()[0]
    wh_id = df_raw.wh_id.unique()[0]

    print('正在数据预处理, 详情: ')
    print('--------------------------------')
    print('事业部Id: {}, 仓库Id: {}, 数据条目: {}\n'.format(bu_id, wh_id, len(df_raw)))

    df_train_raw = df_raw[df_raw.is_train == 1].sort_values('date').reset_index(drop=True)
    df_pred_raw = df_raw[df_raw.is_train == 0].sort_values('date').reset_index(drop=True)

    try:
        df_train_raw, df_pred_raw = preprocess_obj.filter_dropped_cols(df_train_raw, df_pred_raw)
        df_train_raw = preprocess_obj.pro_outstocks(df_train_raw)
        df_train_raw, df_pred_raw = preprocess_obj.align_train_n_test_wh(df_train_raw, df_pred_raw)
        df_train_raw, df_pred_raw = preprocess_obj.fix_temperature(df_train_raw, df_pred_raw)
        df_train_total = preprocess_obj.preprocess(df_train_raw)
        df_train_total, df_pred_raw = preprocess_obj.fix_price(df_train_total, df_pred_raw)
        df_test = preprocess_obj.preprocess(df_pred_raw, False)
        # 获取近30天有售卖的sku
        trained_sku = preprocess_obj.sku_to_train(df_train_total)
        df_train = df_train_total[df_train_total['sku_id'].isin(trained_sku)]
        df_test = df_test[df_test['sku_id'].isin(trained_sku)]
        df_train_total.loc[~df_train_total['sku_id'].isin(trained_sku), 'is_train'] = 4  # 近30天无销量的sku
        # 过滤长尾sku
        df_longtail = preprocess_obj.get_longtail_data(df_train)
        longtail_skus = df_longtail.sku_id.unique()
        df_train_total.loc[df_train_total['sku_id'].isin(longtail_skus), 'is_train'] = 5  # 长尾SKU
    except ValueError:
        print('事业部Id: {}, 仓库Id: {}, 数据条目: {}, 预处理后数据为空！\n'.format(bu_id, wh_id, len(df_raw)))
        return []
    df_all = pd.concat([df_train_total, df_test])
    print('事业部Id: {}, 仓库Id: {}, 数据条目: {}, 预处理成功！\n'.format(bu_id, wh_id, len(df_all)))

    result_list = []
    for val in df_all.values:
        result_list.append(dict(zip(df_all.keys(), val)))

    return result_list


def drop_Outliers(sku_rdd, config):
    """
    过滤异常值
    Args:
        sku_rdd: 原始数据rdd
        config: 配置dict

    Returns:
        过滤异常值后的数据
    """
    preprocess_obj = SaleDataPreprocess(config['cat1_id'], config['forecast_date'])

    df_all = pd.DataFrame(sku_rdd[1], columns=config['buwh_col_names'])
    df_train = df_all[df_all['is_train'] == 1]
    df_test = df_all[df_all['is_train'] == 0]
    df_train = preprocess_obj.drop_Outliers(df_train, config['dropping_upper_bound'], config['dropping_lower_bound'])

    common_sku = set(df_train.sku_id.unique()) & set(df_test.sku_id.unique())
    df_train.loc[~df_train.sku_id.isin(common_sku), 'is_train'] = 6  # 训练/测试集未对齐
    df_test.loc[~df_test.sku_id.isin(common_sku), 'is_train'] = 6  # 训练/测试集未对齐

    df_all = pd.concat([df_train, df_test])
    result_list = []
    for val in df_all.values:
        result_list.append(dict(zip(df_all.keys(), val)))

    return result_list


def features_create(cleaned_rdd, config):
    """
    特征制造函数
    :param cleaned_rdd: spark rdd数据
    :param config: 配置参数字典对象
    :return: 衍生特征spark rdd数据
    """
    df_all = pd.DataFrame(cleaned_rdd[1], columns=config['buwh_col_names'])
    df_all['dt'] = pd.to_datetime(df_all.date)
    bu_id = df_all.bu_id.unique()[0]
    wh_id = df_all.wh_id.unique()[0]
    result_list = []

    df_train_total = df_all[~(df_all.is_train == 0)]
    df_train = df_all[df_all.is_train == 1]
    df_test = df_all[df_all.is_train == 0]

    if (df_train.count()[0] == 0) or (df_test.count()[0] == 0):
        print("{} 事业部 {} 仓 数据异常, 预测结果为空!".format(bu_id, wh_id))
        return result_list

    data = pd.concat([df_train, df_test])

    print('正在特征制造, 详情: ')
    print('--------------------------------')
    print('事业部Id: {}, 仓库Id: {}, 数据条目: {}\n'.format(bu_id, wh_id, len(data)))

    # 派生特征制造
    feats_obj = DerivedFeature(config['cat1_id'])
    try:
        data = feats_obj.create_features(data, df_train_total)
    except:
        print('{} 事业部 {} 仓 特征生产异常！'.format(bu_id, wh_id))
        return []
    print('事业部Id: {}, 仓库Id: {}, 数据条目: {}, 特征制造成功\n'.format(bu_id, wh_id, len(data)))
    # 精度控制
    for col in data.columns:
        _dtype = str(data[col].dtype)
        if 'float' in _dtype:
            data[col] = data[col].map(lambda x: round(x, 5))

    data.drop('dt', axis=1, inplace=True)

    # 构造验证集数据("is_train"标记为3)
    df_train = data[data.is_train == 1]
    for cat in df_train.cat2_name.unique():
        cat_train = df_train[df_train.cat2_name == cat]
        ratio = cat_train.count()[0] * 0.05
        data.loc[df_train.sample(int(ratio)).index, 'is_train'] = 3

    for val in data.values:
        result_list.append(dict(zip(data.keys(), val)))

    return result_list


#############################################
# entry point for PySpark application
#############################################
if __name__ == '__main__':
    main()
