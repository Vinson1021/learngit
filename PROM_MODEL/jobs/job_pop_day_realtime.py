#!/bin/usr/env python3.6
# -*- coding: utf-8 -*-
# ---------------------------------
# ProjectName: data-forecast-spark
# Description: 3P销量预测实时特征生产
# Author: songzhen07
# CreateTime: 2021-04-14 14:41
# ---------------------------------

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dependencies.platform.spark import start_spark
from pyspark.rdd import RDD
from pyspark.sql import SparkSession
import random
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
        app_name='sale_forecast_pop_realtime_feature_engineering')

    # log that main spark job is starting
    log.warn('pop_realtime_feature_engineering is up-and-running')

    # execute spark pipeline
    data, final_config = extract_data(spark, configs)
    final_df = transform_data(spark, log, data, final_config)
    write_data(final_df, final_config)

    # log the success and terminate Spark application
    log.warn('pricing strategy is finished')
    spark.stop()
    return None

#############################################
# extract data
#############################################
def extract_data(spark: SparkSession, configs: dict) -> (RDD, dict):
    """
    extract data from hive

    Args:
        spark: sparkSession
        config: config dict

    Returns:
        raw_data_rdd: data rdd
        config: updated config dict
    """

    from pyspark.sql.types import DateType
    import pyspark.sql.functions as F

    # 日期参数设定
    current_date = datetime.now().strftime("%Y%m%d")
    if 'forecast_date' in configs:
        current_date = configs['forecast_date']
    ## 预测日期
    forecast_date = (pd.to_datetime(current_date) + timedelta(configs['fc_date_delta'])).strftime('%Y%m%d')
    ## 训练开始/结束日期
    train_start_date = (pd.to_datetime(forecast_date) + timedelta(configs['train_duration'])).strftime('%Y%m%d')
    train_end_date = (pd.to_datetime(forecast_date) - timedelta(1)).strftime('%Y%m%d')
    ## 测试结束日期
    forecast_end_date = (pd.to_datetime(forecast_date) + timedelta(configs['test_duration'])).strftime('%Y%m%d')

    extract_sql = """
            select {feature_cols}
             from mart_caterb2b_forecast.app_caterb2b_forecast_sale_3p_rt_input 
             where 
                is_train=1 
                and dt between '{train_start_date}' and '{train_end_date}'
            union all 
            select {feature_cols}
             from mart_caterb2b_forecast.app_caterb2b_forecast_sale_3p_rt_input
             where 
                is_train=0 
                and dt between '{test_start_date}' and '{test_end_date}'    
        """.format(feature_cols=configs['feature_cols'],
                   train_start_date=train_start_date,
                   train_end_date=train_end_date,
                   test_start_date=forecast_date,
                   test_end_date=forecast_end_date
                   )
    # 得到原始数据集
    raw_data = spark.sql(extract_sql).withColumnRenamed('dt', 'date')

    ## 日期字段
    time_func = F.udf(lambda x: datetime.strptime(x, '%Y%m%d'), DateType())
    raw_data = raw_data.withColumn('dt', time_func(F.col('date')))

    new_configs = {
        'col_names': raw_data.schema.names,
        'fc_dt': forecast_date
    }
    configs.update(new_configs)

    return raw_data, configs


def seqOp(x, y):
    """seqFunc"""
    x.append(y)
    return x



def combOp(x, y):
    """combFunc"""
    return x+y


#############################################
# data processing
#############################################
def transform_data(spark, log, raw_data, configs):
    """
    生产特征数据

    :param:
        spark: spark对象
        log：日志打印对象
        _df: 输入spark df数据
        configs： 配置参数字典对象

    :return：
        final_df: 训练+测试特征数据
    """

    import pyspark.sql.functions as F
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer
    from functools import reduce

    # 数据预处理
    raw_rdd = raw_data.rdd.map(lambda row: ((row['wh_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=100, keyfunc=lambda x: x) \
        .flatMap(lambda item: data_prepare(item, configs))
    post_df = spark.createDataFrame(raw_rdd)

    log.info("Update configs by post_df schema...")
    configs.update({'post_col_names': post_df.schema.names})

    # SKU异常值过滤
    tr_df = post_df.filter(F.col('is_train') == 1)
    cleaned_rdd = tr_df.rdd.map(lambda row: ((row['wh_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=100, keyfunc=lambda x: x) \
        .flatMap(lambda item: drop_Outliers(item, configs))
    cleaned_df = spark.createDataFrame(cleaned_rdd)
    te_df = post_df.filter(F.col('is_train') == 0).select(cleaned_df.columns)

    # 保留公共SKU
    cols = ['wh_id', 'bu_id', 'sku_id']
    common_df = cleaned_df.select(cols).join(te_df.select(cols), cols, 'inner').dropDuplicates(cols)
    train_data = cleaned_df.join(common_df, cols, 'inner')
    test_data = te_df.join(common_df, cols, 'inner')
    tl_df = reduce(lambda x, y: x.union(y), [train_data, test_data])

    log.info("Update configs by tl_df schema...")
    configs.update({'cln_col_names': tl_df.schema.names})

    # 特征制造
    total_rdd = tl_df.rdd.map(lambda row: ((row['wh_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=100, keyfunc=lambda x: x) \
        .flatMap(lambda item: features_create(item, configs))
    total_df = spark.createDataFrame(total_rdd)

    # 特征编码
    map_list = ['cat1_name', 'cat2_name', 'day_abbr', 'day_weather', 'night_weather', 'festival_name']
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(total_df) for column in map_list]
    pipeline = Pipeline(stages=indexers)
    final_df = pipeline.fit(total_df).transform(total_df)

    # 删除/重命名特征列
    final_df = final_df.drop(*map_list)
    renamed_list = [col+'_index' for col in map_list]
    mapping = dict(zip(renamed_list, map_list))
    final_df = final_df.select([F.col(c).alias(mapping.get(c, c)) for c in final_df.columns])

    return final_df



def data_prepare(rdd, configs):
    """
    数据预处理函数
    :param buwh_rdd: spark rdd数据
    :param configs: 配置参数字典对象
    :return: 处理后的spark rdd数据
    """
    import pandas as pd
    import numpy as np
    import traceback
    from dependencies.algorithm.sale_forecast_pop.data_process_3p import SalePopDataPreprocess

    pro_obj = SalePopDataPreprocess(configs['fc_dt'])

    values = pd.DataFrame(rdd[1], columns=configs['col_names']).replace('', np.nan)
    wh_id = values.wh_id.unique()[0]
    result_list = list()

    print('正在进行数据预处理, 详情: ')
    print('--------------------------------')
    print('仓库Id: {}, 数据条目: {}'.format(wh_id, values.count()[0]))

    try:
        wh_train = values[values.is_train == 1]
        wh_test = values[values.is_train == 0]

        wh_train = pro_obj.prepro_func(wh_train, is_train=True)
        wh_test = pro_obj.prepro_func(wh_test, is_train=False)

    except Exception as e:
        print('仓库Id: {} 预处理结果为空!'.format(wh_id))
        traceback.print_exc()
        return result_list
    print('仓库Id: {} 处理完成.'.format(wh_id))

    data = pd.concat([wh_train, wh_test])

    for val in data.values:
        result_list.append(dict(zip(data.keys(), val)))

    return result_list



def drop_Outliers(rdd, configs):
    """
    滑动过滤异常值
    :param rdd: spark rdd数据
    :param configs: 配置参数字典对象
    :return: 过滤异常值后的spark rdd数据
    """
    import pandas as pd
    from collections import defaultdict

    group = pd.DataFrame(rdd[1], columns=configs['post_col_names'])
    wh_id = group.wh_id.unique()[0]
    result_list = list()

    print('正在进行异常值过滤, 详情: ')
    print('--------------------------------')
    print('仓库Id: {}, 数据条目: {}'.format(wh_id, group.count()[0]))

    # 过滤重复值
    dfs = group \
        .drop_duplicates(subset=['wh_id', 'bu_id', 'sku_id', 'dt'], keep='first') \
        .reset_index(drop=False)

    _append = lambda a, x, v: {a[i].append(v) for i in x}

    top = 2.5      # 过滤上限
    bottom = 0.2   # 过滤下限


    for key, df_all in dfs.groupby(['wh_id', 'bu_id', 'sku_id']):

        top_set = defaultdict(list)
        bottom_set = defaultdict(list)

        # 节假日期间数据放开过滤
        df_sku = df_all[df_all['festival_name'] == '无']

        # 不对数据量小于7的sku过滤
        if df_sku.count()[0] < 7:
            continue

        data_pro = df_sku[(df_sku.price < df_sku.origin_price) |
                          (df_sku.is_csu_redu == 1)]['total_cnt']
        data_normal = df_sku[~((df_sku.price < df_sku.origin_price) |
                               (df_sku.is_csu_redu == 1))]['total_cnt']

        # 滑动窗口获取异常值索引
        ## 非促销日异常值
        if data_normal.count() >= 7:
            for i in range(7, data_normal.count()+1):
                window = data_normal.iloc[i-7:i]
                avg = window.mean()
                bottom_idx = window[window < avg * bottom].index
                top_idx = window[window > avg * top].index
                _append(top_set, top_idx, avg * top)
                _append(bottom_set, bottom_idx, avg * bottom)
        elif data_normal.count() > 0:
            avg = data_normal.mean()
            bottom_idx = data_normal[data_normal < avg * bottom].index
            top_idx = data_normal[data_normal > avg * top].index
            _append(top_set, top_idx, avg * top)
            _append(bottom_set, bottom_idx, avg * bottom)
        else:
            pass


        ## 促销日异常值
        if data_pro.count() >= 7:
            for j in range(7, data_pro.count()+1):
                window = data_pro.iloc[j-7:j]
                avg = window.mean()
                bottom_idx = window[window < avg * bottom].index
                top_idx = window[window > avg * top].index
                _append(top_set, top_idx, avg * top)
                _append(bottom_set, bottom_idx, avg * bottom)
        elif data_pro.count() > 0:
            avg = data_pro.mean()
            bottom_idx = data_pro[data_pro < avg * bottom].index
            top_idx = data_pro[data_pro > avg * top].index
            _append(top_set, top_idx, avg * top)
            _append(bottom_set, bottom_idx, avg * bottom)
        else:
            pass


        df_top = pd.DataFrame([[i, min(top_set[i])] for i in top_set],
                              columns=['index', 'top_val']).set_index('index')
        df_bottom = pd.DataFrame([[i, max(bottom_set[i])] for i in bottom_set],
                                 columns=['index', 'bottom_val']).set_index('index')


        dfs.loc[df_top.index, 'total_cnt'] = df_top['top_val']
        dfs.loc[df_bottom.index, 'total_cnt'] = df_bottom['bottom_val']

    dfs = dfs[['wh_id', 'bu_id', 'sku_id', 'dt', 'total_cnt']]
    result = group.drop('total_cnt', axis=1).merge(dfs, how='left', on=['wh_id', 'bu_id', 'sku_id', 'dt'])
    result['total_cnt'] = result['total_cnt'].astype(np.float32)

    print('仓库Id: {} 异常值过滤完成.'.format(wh_id))

    for val in result.values:
        result_list.append(dict(zip(result.keys(), val)))

    return result_list



def features_create(rdd, configs):
    """
    特征制造函数
    :param rdd: spark rdd数据
    :param configs: 配置参数字典对象
    :return: 衍生特征spark rdd数据
    """
    import pandas as pd
    from dependencies.algorithm.sale_forecast_pop.feature_create_3p import SalePopFeatureCreate
    feat_obj = SalePopFeatureCreate()

    data = pd.DataFrame(rdd[1], columns=configs['cln_col_names'])
    wh_id = data.wh_id.unique()[0]
    result_list = list()

    print('正在进行特征制造, 详情: ')
    print('--------------------------------')
    print('仓库Id: {}, 数据条目: {}'.format(wh_id, data.count()[0]))
    try:
        # 生产衍生特征
        data = feat_obj.cnt_band_func(data)
        data = feat_obj.seasonal_count(data)
        data = feat_obj.his_sale_avg(data)
        data = feat_obj.get_statistic_features(data)
        data = feat_obj.pro_statistic_features(data)
    except:
        print('{} 仓特征制造异常！'.format(wh_id))
        return result_list
    print('{} 仓特征制造完成'.format(wh_id))

    # 构造验证集数据("is_train"标记为3)
    for h in data.hour.unique():
        df_train = data[(data.is_train == 1) & (data.hour == h)]
        for cat in df_train.cat2_name.unique():
            cat_train = df_train[df_train.cat2_name == cat]
            ratio = cat_train.count()[0] * 0.05
            data.loc[df_train.sample(int(ratio)).index, 'is_train'] = 3

    for val in data.values:
        result_list.append(dict(zip(data.keys(), val)))

    return result_list


#############################################
# output data
#############################################
def write_data(df, configs):
    """Collect data locally and write to CSV.

    :param: rdd to print.
    :return: None
    """
    import pyspark.sql.functions as F

    result_df = df.select(
        F.col('brand_name'),
        F.col('bu_id'),
        F.col('cat1_brand_avg'),
        F.col('cat1_fevl_avg'),
        F.col('cat1_is_cir_count'),
        F.col('cat1_is_csu_count'),
        F.col('cat1_week_avg'),
        F.col('cat2_brand_avg'),
        F.col('cat2_fevl_avg'),
        F.col('cat2_week_avg'),
        F.col('cnt'),
        F.col('cnt_band_area'),
        F.col('cnt_band_area_qpl'),
        F.col('date'),
        F.col('day'),
        F.col('day_temperature'),
        F.col('dt'),
        F.col('his_avg'),
        F.col('hour'),
        F.col('is_cir_redu'),
        F.col('is_csu_redu'),
        F.col('is_holiday'),
        F.col('is_train'),
        F.col('is_weekend'),
        F.col('is_work_day'),
        F.col('itvl_avg'),
        F.col('itvl_ewm_avg'),
        F.col('itvl_roll_avg'),
        F.col('itvl_roll_max'),
        F.col('itvl_roll_min'),
        F.col('month'),
        F.col('night_temperature'),
        F.col('origin_price'),
        F.col('price'),
        F.col('sku_id'),
        F.col('total_cnt'),
        F.col('wh_brand_avg'),
        F.col('wh_fevl_avg'),
        F.col('wh_id'),
        F.col('wh_week_avg'),
        F.col('year'),
        F.col('cat1_name'),
        F.col('cat2_name'),
        F.col('day_abbr'),
        F.col('day_weather'),
        F.col('night_weather'),
        F.col('festival_name'))
    result_df.repartition(10).write.mode('overwrite') \
        .orc(configs['output_path'].format(str(configs['fc_dt'])))
    return None


#############################################
# entry point for PySpark application
#############################################
if __name__ == '__main__':
    main()