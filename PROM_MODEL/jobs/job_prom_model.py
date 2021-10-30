#!/bin/usr/env python3.6
# -*- coding: utf-8 -*-
# ---------------------------------
# ProjectName: data-forecast-spark
# Description: 促销模型-促销回落
# Author: guoyunshen
# CreateTime: 2021-10-23 13:50
# ---------------------------------

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import random
from dependencies.platform.spark import start_spark
from pyspark.rdd import RDD
from pyspark.sql import SparkSession

random.seed(42)
np.random.seed(42)


#############################################
# main function
#############################################
def main():
    """
    Main script definition.

    :return: None
    """
    # start Spark application and get Spark session, logger and config
    spark, log, configs = start_spark(
        app_name='sale_forecast_prom_fallback')

    # log that main spark job is starting
    log.warn('sale_forecast_prom_fallback is up-and-running')

    # execute spark pipeline
    final_config, train_data, test_data = extract_data(spark, configs)
    final_df, final_config = transform_data(spark, log, train_data, test_data, final_config)
    write_data(final_df, final_config)

    # log the success and terminate Spark application
    log.warn('sale_forecast_prom_fallback is finished')
    spark.stop()
    return None


#############################################
# extract data
#############################################
def extract_data(spark: SparkSession, configs: dict) -> (RDD, dict):
    """
    Load data from hive table.

    :param:
        spark: spark对象
        configs：配置参数

    :return:
        configs, train_df, test_df
    """

    # 日期参数设定
    current_date = datetime.now().strftime("%Y%m%d")
    if 'forecast_date' in configs:
        current_date = configs['forecast_date']

    ## 预测日期
    forecast_date = (pd.to_datetime(current_date) + timedelta(configs['fc_date_delta'])).strftime('%Y%m%d')

    ## 训练开始/结束日期
    train_start_date = (pd.to_datetime(forecast_date) + timedelta(configs['train_duration'])).strftime('%Y%m%d')
    train_end_date = (pd.to_datetime(forecast_date) - timedelta(1)).strftime('%Y%m%d')

    ## 筛选 case 的开始日期
    start_date = (pd.to_datetime(train_end_date) - timedelta(configs['case_duration'])).strftime('%Y%m%d')

    ## 测试结束日期
    forecast_end_date = (pd.to_datetime(forecast_date) + timedelta(configs['test_duration'])).strftime('%Y%m%d')

    train_sql = '''
            SELECT n.*
              FROM (
                    SELECT *
                      FROM (
                            SELECT *,
                                   avg(arranged_cnt) over(partition by wh_id, sku_id order by dt rows between 2 preceding and 0 preceding) as last_3days_mean,
                                   avg(arranged_cnt) over(partition by wh_id, sku_id order by dt rows between 9 preceding and 3 preceding) as fore_7days_mean,
                                   avg(ape_T0) over(partition by wh_id, sku_id order by dt rows between 2 preceding and 0 preceding) as last_3days_mean_apeT0,
                                   avg(ape_T0) over(partition by wh_id, sku_id order by dt rows between 9 preceding and 3 preceding) as fore_7days_mean_apeT0,
                                   COUNT(sku_id) OVER (PARTITION BY wh_id, sku_id) as row_num,
                                   ROW_NUMBER() OVER (PARTITION BY wh_id, sku_id ORDER BY dt desc) as latest
                              FROM (
                                    SELECT *,
                                           activity_reduction_cnt+activity_seckill_cnt+activity_promotion_cnt as all_pro_num,
                                           case when (arranged_cnt = 0 and pred_t0 >= 1) then pred_t0
                                                when (arranged_cnt = 0 and pred_t0 < 1)  then 0
                                                else abs(pred_t0 - arranged_cnt) / arranged_cnt
                                                 end as ape_T0
                                      FROM mart_caterb2b_forecast.app_forecast_sale_badcase_general_analysis
                                     where dt BETWEEN {} and {}
                                       and cat1_name = '肉禽水产（冷冻）'
                                       and on_shelf = 1
                                       and (able_sell = 1 or arranged_cnt > 0)
                                   )
                           )
                     where ape_T0 is not NULL
                       and row_num > 10
                       and latest = 1
                       and last_3days_mean_apeT0 >= 0.5
                       and 0.5 * (fore_7days_mean+1) > (last_3days_mean+1)
                   ) m
              join (
                    SELECT *
                      FROM mart_caterb2b_forecast.app_caterb2b_forecast_sale_input_v3
                     where dt BETWEEN  {} and {}
                       and cat1_name = '肉禽水产（冷冻）'
                       and is_train = 0
                   ) n
                on m.bu_id = n.bu_id
               and m.wh_id = n.wh_id
               and m.sku_id = n.sku_id
    '''

    test_sql = '''
            SELECT n.*
              FROM (
                    SELECT *
                      FROM (
                            SELECT *,
                                   avg(arranged_cnt) over(partition by wh_id, sku_id order by dt rows between 2 preceding and 0 preceding) as last_3days_mean,
                                   avg(arranged_cnt) over(partition by wh_id, sku_id order by dt rows between 9 preceding and 3 preceding) as fore_7days_mean,
                                   avg(ape_T0) over(partition by wh_id, sku_id order by dt rows between 2 preceding and 0 preceding) as last_3days_mean_apeT0,
                                   avg(ape_T0) over(partition by wh_id, sku_id order by dt rows between 9 preceding and 3 preceding) as fore_7days_mean_apeT0,
                                   COUNT(sku_id) OVER (PARTITION BY wh_id, sku_id) as row_num,
                                   ROW_NUMBER() OVER (PARTITION BY wh_id, sku_id ORDER BY dt desc) as latest
                              FROM (
                                    SELECT *,
                                           activity_reduction_cnt+activity_seckill_cnt+activity_promotion_cnt as all_pro_num,
                                           case when (arranged_cnt = 0 and pred_t0 >= 1) then pred_t0
                                                when (arranged_cnt = 0 and pred_t0 < 1)  then 0
                                                else abs(pred_t0 - arranged_cnt) / arranged_cnt
                                                 end as ape_T0
                                      FROM mart_caterb2b_forecast.app_forecast_sale_badcase_general_analysis
                                     where dt BETWEEN {} and {}
                                       and cat1_name = '肉禽水产（冷冻）'
                                       and on_shelf = 1
                                       and (able_sell = 1 or arranged_cnt > 0)
                                   )
                           )
                     where ape_T0 is not NULL
                       and row_num > 10
                       and latest = 1
                       and last_3days_mean_apeT0 >= 0.5
                       and 0.5 * (fore_7days_mean+1) > (last_3days_mean+1)
                   ) m
              join (
                    SELECT *
                      FROM (
                            SELECT *,
                                   COUNT(sku_id) OVER (PARTITION BY wh_id, sku_id) as row_num
                              FROM mart_caterb2b_forecast.app_caterb2b_forecast_sale_input_v3
                             where dt BETWEEN  {} and {}
                               and cat1_name = '肉禽水产（冷冻）'
                               and is_train = 1
                           ) da
                     where da.row_num >= 60
                   ) n
                on m.bu_id = n.bu_id
               and m.wh_id = n.wh_id
               and m.sku_id = n.sku_id
              '''

    # 得到原始数据集
    train_data = spark.sql(train_sql.format(start_date, train_end_date,
                                            start_date, train_end_date,
                                            train_start_date, train_end_date))

    test_data = spark.sql(test_sql.format(start_date, train_end_date,
                                          start_date, train_end_date,
                                          forecast_date, forecast_end_date))

    ## 获取列名
    bd_train_data_col_names = train_data.schema.names
    bd_test_data_col_names = test_data.schema.names

    ## 更新配置参数字典
    append_conf = {"current_date": current_date,
                   "forecast_date": forecast_date,
                   "train_start_date": train_start_date,
                   "train_end_date": train_end_date,
                   "start_date": start_date,
                   "forecast_end_date": forecast_end_date,
                   "bd_train_data_col_names": bd_train_data_col_names,
                   "bd_test_data_col_names": bd_test_data_col_names
                   }
    configs.update(append_conf)

    return configs, train_data, test_data


def seqOp(x, y):
    """
    seqFunc.
    """
    x.append(y)
    return x


def combOp(x, y):
    """
    combFunc.
    """
    return x+y


###########################################################
# data processing
###########################################################
def transform_data(spark, log, train_data, test_data, configs):
    """
    生成最终预测结果数据

    :param:
        spark: spark对象
        log：日志打印对象
        train_data, test_data: 输入spark df 数据
        configs： 配置参数字典对象

    :return：
        final_df: 预测结果数据
    """

    import pyspark.sql.functions as F
    from pyspark.sql import Window

    # 数据预处理
    log.info("Start to Preprocess...")

    # 将 train_data 和 test_data 整合在一起处理
    all_data = train_data.drop('row_num').union(test_data)

    # 选定需要的 columns
    cols = ['bu_id', 'wh_id', 'sku_id', 'cat1_name', 'cat2_name',
            'seq_num', 'redu_num', 'csu_redu_num', 'cir_redu_num',
            'pro_num', 'day_abbr', 'is_work_day', 'is_weekend',
            'is_holiday', 'is_train', 'arranged_cnt', 'dt']

    all_data = all_data.select(*cols)

    # 对促销特征列进行空值填充
    num_cols = ['seq_num', 'redu_num', 'csu_redu_num', 'cir_redu_num', 'pro_num']
    all_data = all_data.fillna(0, subset=num_cols)

    # 统计当天总体的销量个数
    all_data = all_data.withColumn('all_pro_num', (all_data['seq_num'] + all_data['redu_num'] + all_data['pro_num']))

    all_data = all_data.sort(F.col('bu_id'), F.col('wh_id'),
                             F.col('sku_id'), F.col('dt'))

    # 更新配置参数字典
    append_conf = {
        "bd_all_data_col_names": all_data.schema.names
                   }
    configs.update(append_conf)

    log.info("Start Feature Engineering...")
    data_rdd = all_data.rdd.map(lambda row: ((row['bu_id'], row['wh_id'], row['sku_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=100, keyfunc=lambda x: x) \
        .flatMap(lambda item: arr_feature(item, configs))

    data_df = spark.createDataFrame(data_rdd)

    # 促销回落 case list [‘bu + wh + sku’ 维度]
    sku_list_rdd = data_df.rdd.map(
        lambda row: ((row['bu_id'], row['wh_id'], row['sku_id']), 1)) \
        .groupByKey().mapValues(len)
    sku_list = sku_list_rdd.keys().collect()

    log.info('促销回落case的个数为: ' + str(len(sku_list)))

    # 更新配置参数字典
    append_conf = {"bd_data_df_col_names": data_df.schema.names
                   }
    configs.update(append_conf)

    log.info("Start to forecast...")

    # 剔除不满35天的sku
    forecast_rdd = data_df.rdd.map(lambda row: ((row['bu_id'], row['wh_id'], row['sku_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=100, keyfunc=lambda x: x) \
        .flatMap(lambda item: forecast_func(item, configs))

    forecast_df = spark.createDataFrame(forecast_rdd)

    results_df = spark.createDataFrame(results_rdd) \
                      .where(F.col('is_train') == 0) \
                      .sort(F.col('bu_id'), F.col('wh_id'), F.col('sku_id'), F.col('sold_days')) \
                      .select(F.col('bu_id'), F.col('wh_id'), F.col('sku_id'),
                              F.col('dt'), F.col('predictions'))

    ## 将预测结果整合成json的形式
    ## 加入 ‘total_amt_fc’ 字段
    sum_win = Window.partitionBy('sku_id', 'wh_id', 'bu_id')

    agg_df = results_df.withColumn('total_amt_fc', F.sum('predictions').over(sum_win)) \
                       .groupby('sku_id', 'wh_id', 'bu_id', 'total_amt_fc') \
                       .pivot('dt').agg(F.first('predictions').alias('pred'))

    final_results_df = agg_df.withColumn('daily_amt_fc', F.regexp_replace(
        F.to_json(F.struct([F.when(F.lit(x).like('20%'), agg_df[x]).alias(x) for x in agg_df.columns])), 'pred', '')) \
        .select(F.col('bu_id'), F.col('wh_id'), F.col('sku_id'),
                F.col('total_amt_fc'), F.col('daily_amt_fc')) \
        .withColumn('fc_dt', F.lit(configs['forecast_date']))

    return final_results_df, configs


###########################################################
# Feature engineering about real_arr
###########################################################
def arr_feature(all_df, configs):
    """
    生产销量特征
    :param all_df: spark rdd数据
    :param configs: 配置参数字典对象
    :return: 训练集 & 测试集的spark rdd数据
    """

    import pandas as pd
    import numpy as np
    from dependencies.algorithm.prom_model.promodel_feature_util import FeatureUtil

    sku_data = pd.DataFrame(all_df[1], columns=configs['bd_all_data_col_names']) \
                 .replace('', np.nan)

    # 剔除训练数据小于 60天 的 sku.
    if len(sku_data[sku_data.is_train == 1]) < 60:
        return []

    wh_id, sku_id = int(sku_data.wh_id.unique()[0]), int(sku_data.sku_id.unique()[0])
    sku_data = sku_data.sort_values('dt')
    feature_obj = FeatureUtil()

    # 获取所有的真实销量数据
    arr_list = sku_data[sku_data.is_train == 1].arranged_cnt
    roll_7d_avg = arr_list.rolling(30, min_periods=30).apply(feature_obj.clc_fc_val).dropna()

    avg_7d_0 = arr_list.rolling(30, min_periods=30).apply(lambda x: feature_obj.window_avg_arr(x, 0)).dropna()
    avg_7d_1 = arr_list.rolling(30, min_periods=30).apply(lambda x: feature_obj.window_avg_arr(x, 1)).dropna()
    avg_7d_2 = arr_list.rolling(30, min_periods=30).apply(lambda x: feature_obj.window_avg_arr(x, 2)).dropna()
    avg_7d_3 = arr_list.rolling(30, min_periods=30).apply(lambda x: feature_obj.window_avg_arr(x, 3)).dropna()

    avg_3d_0 = arr_list.rolling(30, min_periods=30).apply(lambda x: feature_obj.window_avg_arr(x, 4)).dropna()
    avg_last7d = arr_list.rolling(30, min_periods=30).apply(lambda x: feature_obj.latest_val(x)).dropna()

    # 销量特征数据
    arr_and_avg  = roll_7d_avg.tolist()
    arr_and_avg0 = avg_7d_0.tolist()
    arr_and_avg1 = avg_7d_1.tolist()
    arr_and_avg2 = avg_7d_2.tolist()
    arr_and_avg3 = avg_7d_3.tolist()
    arr_and_avg4 = avg_3d_0.tolist()
    arr_and_avg5 = avg_last7d.tolist()

    # 外推 list
    wt = arr_list.tolist()
    func = lambda x, y: x * y
    lst = [0.2, 0.3, 0.5]

    # 测试集数据外推到指定的天数
    for i in range(6):
        wt.append(sum(map(func, wt[-3:], lst)))

        arr_and_avg.append(feature_obj.clc_fc_val(pd.Series(wt[-30:])))
        arr_and_avg0.append(feature_obj.window_avg_arr(pd.Series(wt[-30:]), 0))
        arr_and_avg1.append(feature_obj.window_avg_arr(pd.Series(wt[-30:]), 1))
        arr_and_avg2.append(feature_obj.window_avg_arr(pd.Series(wt[-30:]), 2))
        arr_and_avg3.append(feature_obj.window_avg_arr(pd.Series(wt[-30:]), 3))
        arr_and_avg4.append(feature_obj.window_avg_arr(pd.Series(wt[-30:]), 4))
        arr_and_avg5.append(feature_obj.latest_val(pd.Series(wt[-30:])))

    # 剔除多余的数据

    try:
        sku_data = sku_data[30:]

        sku_data['avg_7d_0'] = arr_and_avg0
        sku_data['avg_7d_1'] = arr_and_avg1
        sku_data['avg_7d_2'] = arr_and_avg2
        sku_data['avg_7d_3'] = arr_and_avg3
        sku_data['roll_7d_avg'] = arr_and_avg

        sku_data['avg_3d_0'] = arr_and_avg4
        sku_data['avg_last7d'] = arr_and_avg5

    except:
        print('wh_id:{}  sku_id:{}  特征生产异常！'.format(wh_id, sku_id))
        return []

    # 生成指定的 result_list
    result_list = []
    for val in sku_data.values:
        result_list.append(dict(zip(sku_data.keys(), val)))

    return result_list


###########################################################
# Forecast Function
###########################################################
def forecast_func(feature_rdd, configs):
    """
    矩阵计算
    :param feature_rdd: spark rdd数据
    :param configs: 配置参数字典对象
    :return: 矩阵计算结果的spark rdd数据
    """

    import pandas as pd
    import numpy as np
    from sklearn import preprocessing
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import LinearRegression

    sku_data = pd.DataFrame(feature_rdd[1], columns=configs['bd_data_df_col_names']) \
        .replace('', np.nan)

    # 保存 sku 的基本信息
    bu_id = int(sku_data.bu_id.unique()[0])
    wh_id, sku_id = int(sku_data.wh_id.unique()[0]), int(sku_data.sku_id.unique()[0])
    sku_data = sku_data.sort_values('dt')
    pred_dates = sku_data[sku_data.is_train == 0].dt
    avg_result = sku_data[sku_data.is_train == 0].roll_7d_avg

    # 选择使用的特征列
    feat_cols = ['dt', 'arranged_cnt', 'is_train', 'all_pro_num', 'seq_num',
                 'pro_num', 'redu_num', 'csu_redu_num', 'cir_redu_num',
                 'roll_7d_avg', 'avg_7d_0', 'avg_3d_0']

    sku_data = sku_data[feat_cols]

    # 去除不需要的特征列
    train_data = sku_data[sku_data.is_train == 1].drop(['dt', 'is_train'], axis=1)
    test_data = sku_data[sku_data.is_train == 0].drop(['dt', 'is_train'], axis=1)

    # 标准化处理
    train_data_y, train_data_X = train_data.iloc[:, 0], train_data.iloc[:, 1:]
    test_data_y, test_data_X = test_data.iloc[:, 0], test_data.iloc[:, 1:]

    scaler = preprocessing.StandardScaler()
    scaler.fit(train_data_X)
    train_data_X_std = scaler.transform(train_data_X)
    test_data_X_std = scaler.transform(test_data_X)

    # 预测部分
    clf = Ridge()
    clf.fit(train_data_X_std, train_data_y)

    pred = clf.predict(test_data_X_std)
    pred = list(map(lambda x: max(x, 0.), pred))

    result = pd.DataFrame({
                           'bu_id': [bu_id]*len(pred_dates),
                           'wh_id': [wh_id]*len(pred_dates),
                           'sku_id': [sku_id]*len(pred_dates),
                           'date': pred_dates,
                           'results': avg_result,
                           'predictions': pred})

    result_list = []
    for val in result.values:
        result_list.append(dict(zip(result.keys(), val)))

    return result_list


#############################################
# output data
#############################################
def write_data(df, configs):
    """
    Collect data locally and write to CSV.

    :param: rdd to print.
    :return: None
    """
    import pyspark.sql.functions as F

    result_df = df.select(
        F.col('bu_id'),
        F.col('wh_id'),
        F.col('sku_id'),
        F.col('total_amt_fc'),
        F.col('daily_amt_fc'))

    result_df.repartition(10).write.mode('overwrite') \
        .orc(configs['output_path'].format(str(configs['forecast_date'])))

    return None


#############################################
# Entry point for PySpark Application
#############################################
if __name__ == '__main__':
    main()
