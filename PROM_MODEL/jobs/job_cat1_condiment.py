#!/bin/usr/env python3.6
# -*- coding: utf-8 -*-
# ---------------------------------
# ProjectName: data-forecast-spark
# Description: 干调品类模型
# Author: songzhen07
# CreateTime: 2021-02-23 17:34
# ---------------------------------

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dependencies.platform.spark import start_spark
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
        app_name='sale_forecast_cat1_condiment_feature_engineering')

    # log that main spark job is starting
    log.warn('cat1_condiment_feature_engineering is up-and-running')

    # execute spark pipeline
    final_config, data = extract_data(spark, configs)
    final_df = transform_data(spark, log, data, final_config)
    write_data(final_df, final_config)

    # log the success and terminate Spark application
    log.warn('pricing strategy is finished')
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
    ## 预测日期
    forecast_date = (pd.to_datetime(current_date) + timedelta(configs['fc_date_delta'])).strftime('%Y%m%d')
    ## 训练开始/结束日期
    train_start_date = (pd.to_datetime(forecast_date) + timedelta(configs['train_duration'])).strftime('%Y%m%d')
    train_end_date = (pd.to_datetime(forecast_date) - timedelta(1)).strftime('%Y%m%d')
    ## 测试结束日期
    forecast_end_date = (pd.to_datetime(forecast_date) + timedelta(configs['test_duration'])).strftime('%Y%m%d')

    inquire_sql = '''
        select v.is_train,v.bu_id,v.wh_id,v.sku_id,v.cat1_name,v.cat2_name,v.cnt_band,v.brand_name,v.tax_rate,v.csu_origin_price,
               v.seq_num,v.seq_price,v.csu_redu_num,v.csu_redu_mdct,v.csu_redu_adct,v.cir_redu_num,v.cir_redu_mdct,v.cir_redu_adct,
               v.pro_num,v.discount_price,v.day_abbr,v.is_work_day,v.is_weekend,v.is_holiday,v.festival_name,v.western_festival_name,
               v.weather,v.weather_b,v.avg_t,v.avg_t_b,v.new_byrs,v.old_byrs,v.poultry_byrs,v.vege_fruit_byrs,v.egg_aquatic_byrs,
               v.standard_byrs,v.vege_byrs,v.meet_byrs,v.frozen_meat_byrs,v.forzen_semi_byrs,v.wine_byrs,v.rice_byrs,v.egg_byrs,
               v.is_on_shelf,v.is_outstock,v.arranged_cnt,v.dt
         from mart_caterb2b_forecast.app_caterb2b_forecast_sale_input_v3 v
         join mart_caterb2b.dim_caterb2b_warehouse w
           on v.wh_id = w.id
        where v.dt between {} and {}
          and v.dt not between '20200201' and '20200415'
          and v.is_train = 1
          and v.cat1_id = {}
        union all 
        select v.is_train,v.bu_id,v.wh_id,v.sku_id,v.cat1_name,v.cat2_name,v.cnt_band,v.brand_name,v.tax_rate,v.csu_origin_price,
               v.seq_num,v.seq_price,v.csu_redu_num,v.csu_redu_mdct,v.csu_redu_adct,v.cir_redu_num,v.cir_redu_mdct,v.cir_redu_adct,
               v.pro_num,v.discount_price,v.day_abbr,v.is_work_day,v.is_weekend,v.is_holiday,v.festival_name,v.western_festival_name,
               v.weather,v.weather_b,v.avg_t,v.avg_t_b,v.new_byrs,v.old_byrs,v.poultry_byrs,v.vege_fruit_byrs,v.egg_aquatic_byrs,
               v.standard_byrs,v.vege_byrs,v.meet_byrs,v.frozen_meat_byrs,v.forzen_semi_byrs,v.wine_byrs,v.rice_byrs,v.egg_byrs,
               v.is_on_shelf,v.is_outstock,v.arranged_cnt,v.dt 
         from mart_caterb2b_forecast.app_caterb2b_forecast_sale_input_v3 v
         join mart_caterb2b.dim_caterb2b_warehouse w
           on v.wh_id = w.id
        where v.dt between {} and {}
          and v.is_train = 0
          and v.cat1_id = {}
    '''
    # 得到原始数据集
    raw_data = spark.sql(inquire_sql.format(train_start_date, train_end_date,
                                            configs['cat1_id'],
                                            forecast_date, forecast_end_date,
                                            configs['cat1_id'])).withColumnRenamed('dt', 'date')

    bd_col_names = raw_data.schema.names

    ## 更新配置参数字典
    append_conf = {"current_date": current_date,
                   "forecast_date": forecast_date,
                   "train_start_date": train_start_date,
                   "train_end_date": train_end_date,
                   "forecast_end_date": forecast_end_date,
                   "bd_col_names": bd_col_names
                   }
    configs.update(append_conf)

    return configs, raw_data


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
def transform_data(spark, log, _df, configs):
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

    # 数据预处理
    log.info("Start to preprocess...")
    buwh_rdd = _df.rdd.map(lambda row: ((row['bu_id'], row['wh_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=100, keyfunc=lambda x: x) \
        .flatMap(lambda item: data_preprocess(item, configs))
    buwh_df = spark.createDataFrame(buwh_rdd)

    log.info("Update configs by buwh schema...")
    buwh_col_names = buwh_df.schema.names
    configs.update({"buwh_col_names": buwh_col_names})

    # 滑动过滤异常值
    log.info("Start filter outliers by windows operation...")
    cleaned_rdd = buwh_df.rdd.map(lambda row: ((row['bu_id'], row['wh_id'], row['sku_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=1000, keyfunc=lambda x: x) \
        .flatMap(lambda item: drop_Outliers(item, configs))
    cleaned_df = spark.createDataFrame(cleaned_rdd)

    # 衍生特征制造
    feature_rdd = cleaned_df.rdd.map(lambda row: ((row['bu_id'], row['wh_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=100, keyfunc=lambda x: x) \
        .flatMap(lambda item: features_create(item, configs))
    feature_df = spark.createDataFrame(feature_rdd)

    # 特征编码
    map_list = ['cat2_name', 'day_abbr', 'festival_name', 'western_festival_name', 'weather', 'weather_b', 'year']
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(feature_df) for column in map_list]
    pipeline = Pipeline(stages=indexers)
    final_df = pipeline.fit(feature_df).transform(feature_df)

    # 删除/重命名特征列
    final_df = final_df.drop(*map_list)
    renamed_list = [col+'_index' for col in map_list]
    mapping = dict(zip(renamed_list, map_list))
    final_df = final_df.select([F.col(c).alias(mapping.get(c, c)) for c in final_df.columns])

    # 添加日期分区字段
    #final_df = final_df.withColumn('dt', F.lit(configs['forecast_date']))

    return final_df



def data_preprocess(buwh_rdd, configs):
    """
    数据预处理函数
    :param buwh_rdd: spark rdd数据
    :param configs: 配置参数字典对象
    :return: 处理后的spark rdd数据
    """
    import pandas as pd
    import numpy as np
    from dependencies.algorithm.cat_model.cat1_condiment_data_process import SaleDataPreprocess

    # 转换为pandas DataFrame
    df_raw = pd.DataFrame(buwh_rdd[1], columns=configs['bd_col_names']) \
        .replace('', np.nan)
    df_raw['dt'] = pd.to_datetime(df_raw['date'])
    bu_id = df_raw.bu_id.unique()[0]
    wh_id = df_raw.wh_id.unique()[0]

    print('正在数据预处理, 详情: ')
    print('--------------------------------')
    print('事业部Id: {}, 仓库Id: {}, 数据条目: {}\n'.format(bu_id, wh_id, len(df_raw)))

    df_train_raw = df_raw[df_raw.is_train == 1].sort_values('date').reset_index(drop=True)
    df_pred_raw = df_raw[df_raw.is_train == 0].sort_values('date').reset_index(drop=True)

    # 数据预处理
    ########################################
    ## 过滤测试数据不足20条的sku
    sku_size = df_pred_raw.groupby('sku_id').size()
    less_20d_skus = sku_size[sku_size < 20].index
    df_pred_raw = df_pred_raw[~df_pred_raw.sku_id.isin(less_20d_skus)]

    try:
        pro_obj = SaleDataPreprocess(configs['forecast_date'])
        ## 缺货日期数据修正
        df_train_raw = pro_obj.pro_outstocks(df_train_raw)

        ## 测试集T0温度修正
        train_last_dt = df_train_raw.sort_values('dt').dt.iloc[-1]
        train_last_temp = df_train_raw.sort_values('dt').avg_t.iloc[-1]
        if (train_last_dt == pd.to_datetime(configs['forecast_date']) - timedelta(1)):
            if np.isnan(train_last_temp) == False:
                df_pred_raw.loc[df_pred_raw.dt == pd.to_datetime(configs['forecast_date']), 'avg_t_b'] = train_last_temp

        ## 预处理训练集
        df_train_total = pro_obj.preprocess(df_train_raw, is_train=True)

        ## 替换测试集价格
        for sku, sku_data in df_train_total.groupby('sku_id'):
            last_price = sku_data.sort_values('dt').csu_origin_price.iloc[-1]
            if sku not in df_pred_raw.sku_id.unique():
                continue
            test_price = df_pred_raw[df_pred_raw.sku_id == sku]['csu_origin_price'].unique()[0]
            if test_price != last_price:
                df_pred_raw.loc[df_pred_raw.sku_id == sku, 'csu_origin_price'] = last_price

        ## 预处理测试集
        df_test = pro_obj.preprocess(df_pred_raw, is_train=False)

        ## 标记近30天未售卖的sku
        trained_sku = pro_obj.sku_to_train(df_train_total)
        df_train = df_train_total[df_train_total['sku_id'].isin(trained_sku)]
        df_test = df_test[df_test['sku_id'].isin(trained_sku)]
        df_train_total.loc[~df_train_total['sku_id'].isin(trained_sku), 'is_train'] = 1024

        ## (从训练集)获取并标记长尾数据
        df_longtail = pro_obj.get_longtail_data(df_train)
        longtail_skus = df_longtail.sku_id.unique()
        df_train_total.loc[df_train_total['sku_id'].isin(longtail_skus), 'is_train'] = 1024
    except ValueError:
        print('{} 事业部 {} 仓 数据预处理结果为空！'.format(bu_id, wh_id))
        return []
    print('{} 事业部 {} 仓 数据预处理已完成！'.format(bu_id, wh_id))

    df_all = pd.concat([df_train_total, df_test])

    result_list = []
    for val in df_all.values:
        result_list.append(dict(zip(df_all.keys(), val)))

    return result_list



def drop_Outliers(sku_rdd, configs):
    """
    滑动过滤异常值
    :param sku_rdd: spark rdd数据
    :param configs: 配置参数字典对象
    :return: 过滤异常值后的spark rdd数据
    """
    import pandas as pd
    from collections import defaultdict

    df_all = pd.DataFrame(sku_rdd[1], columns=configs['buwh_col_names'])

    keys = df_all.keys()
    df_sku = df_all[df_all.is_train == 1]
    top_idx = []
    top_set = defaultdict(list)
    bottom_idx = []
    result_list = []

    _append = lambda a, x, v: {a[i].append(v) for i in x}

    top = 3.0      # 过滤上限
    bottom = 0.2   # 过滤下限

    # 节假日期间数据放开过滤
    df_sku = df_sku[df_sku['festival_name'] == '无']

    # 不对数据量小于7的sku过滤
    if df_sku.count()[0] < 7:
        for val in df_all.values:
            result_list.append(dict(zip(keys, val)))
        return result_list

    # 区分促销/非促销数据
    ## 非促销日期数据
    data_normal = df_sku[(df_sku['pro_num'] == 0) &
                         (df_sku['seq_num'] == 0) &
                         (df_sku['csu_redu_num'] == 0)]['arranged_cnt']

    ## 促销日期数据
    data_pro = df_sku[(df_sku['pro_num'] != 0) |
                      (df_sku['seq_num'] != 0) |
                      (df_sku['csu_redu_num'] != 0)]['arranged_cnt']

    # 滑动窗口获取异常值索引
    ## 非促销日异常值
    if data_normal.count() >= 7:
        for i in range(7, data_normal.count()+1):
            window = data_normal.iloc[i-7:i]
            avg = window.mean()
            bottom_idx.extend(window[window < avg * bottom].index)
            top_idx = window[window > avg * top].index
            _append(top_set, top_idx, avg * top)
    else:
        avg = data_normal.mean()
        bottom_idx.extend(data_normal[data_normal < avg * bottom].index)
        top_idx = data_normal[data_normal > avg * top].index
        _append(top_set, top_idx, avg * top)

    ## 促销日异常值
    if data_pro.count() >= 7:
        for j in range(7, data_pro.count()+1):
            window = data_pro.iloc[j-7:j]
            avg = window.mean()
            bottom_idx.extend(window[window < avg * bottom].index)
            top_idx = window[window > avg * top].index
            _append(top_set, top_idx, avg * top)
    else:
        avg = data_pro.mean()
        bottom_idx.extend(data_pro[data_pro < avg * bottom].index)
        top_idx = data_pro[data_pro > avg * top].index
        _append(top_set, top_idx, avg * top)

    # 保留全部数据，更改需要剔除数据的‘is_train’标签
    # 对超过上线的销量进行替换
    df_all.loc[df_all.index.isin(bottom_idx), 'is_train'] = 1024
    df_alter = pd.DataFrame([[i, max(top_set[i])] for i in top_set], columns=['index', 'alt_val']).set_index('index')
    df_all.loc[df_alter.index, 'arranged_cnt'] = df_alter['alt_val']

    for val in df_all.values:
        result_list.append(dict(zip(keys, val)))

    return result_list



def features_create(cleaned_rdd, configs):
    """
    特征制造函数
    :param cleaned_rdd: spark rdd数据
    :param configs: 配置参数字典对象
    :return: 衍生特征spark rdd数据
    """
    import pandas as pd
    from dependencies.algorithm.cat_model.cat1_condiment_derived_features import SaleStatisticalFeatures

    df_all = pd.DataFrame(cleaned_rdd[1], columns=configs['buwh_col_names'])
    df_all['dt'] = pd.to_datetime(df_all.date)
    bu_id = df_all.bu_id.unique()[0]
    wh_id = df_all.wh_id.unique()[0]
    result_list = []

    df_train = df_all[df_all.is_train == 1]
    df_test = df_all[df_all.is_train == 0]

    # 确保训练集&测试集的sku对应
    common_sku = set(df_train.sku_id.unique()) & set(df_test.sku_id.unique())
    df_train = df_train[df_train.sku_id.isin(common_sku)]
    df_test = df_test[df_test.sku_id.isin(common_sku)]

    if (df_train.count()[0] == 0) or (df_test.count()[0] == 0):
        print("{} 事业部 {} 仓 数据异常, 预测结果为空!".format(bu_id, wh_id))
        return result_list

    data = pd.concat([df_train, df_test])
    all_train_data = df_all[df_all.is_train.isin([1, 1024])]

    print('正在特征制造, 详情: ')
    print('--------------------------------')
    print('事业部Id: {}, 仓库Id: {}, 数据条目: {}\n'.format(bu_id, wh_id, len(data)))

    # 派生特征制造
    feature_obj = SaleStatisticalFeatures()
    try:
        data = feature_obj.get_week_sale_ratio(data, all_train_data)
        data = feature_obj.cnt_band_func(data)
        data = feature_obj.seasonal_count(data, all_train_data)
        data = feature_obj.his_sale_avg(data)
        data = feature_obj.his_pro_sale_avg(data)
        data = feature_obj.same_cat_rebate(data)
        data = feature_obj.get_statistic_features(data)
        data = feature_obj.pro_statistic_features(data)
        byr_cols = [col for col in data.columns if '_byrs' in col]
        data = feature_obj.byrs_count(data, all_train_data, col_list=byr_cols)
    except:
        print('{} 事业部 {} 仓 特征生产异常！'.format(bu_id, wh_id))
        return []

    # 精度控制
    for col in data.columns:
        _dtype = str(data[col].dtype)
        if 'float' in _dtype:
            data[col] = data[col].map(lambda x: round(x, 5))

    data.drop('dt', axis=1, inplace=True)

    #     # 数据量控制(避免图灵报错)
    #     if (data[data.is_train == 1].count()[0] < 2000) or \
    #             (data[data.is_train == 0].count()[0] < 5):
    #         return []

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
# output data
#############################################
def write_data(df, configs):
    """Collect data locally and write to CSV.

    :param: rdd to print.
    :return: None
    """
    df.repartition(10).write.mode('overwrite') \
        .orc(configs['output_path'].format(str(configs['forecast_date'])))
    return None


#############################################
# entry point for PySpark application
#############################################
if __name__ == '__main__':
    main()