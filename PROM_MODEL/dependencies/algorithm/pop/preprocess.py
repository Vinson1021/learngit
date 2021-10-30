# coding: utf-8

import getopt
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from os import listdir
from os.path import isfile, join

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.functions import *
import numpy as np
from functools import reduce
import os, sys
from itertools import chain
from pyspark.sql.types import *
from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.sql import Window

import numpy as np
import pandas as pd


# label_start_date = '20210218' # 有数据的日期
# forecast_date = '20210318' # 第一周数据异常较多
# 如何支持多日期预测
def extract_data_old(spark, configs):
    '''
    laoding all data for train and predict
    '''
    if 'label_start_date' in configs:
        label_start_date = configs['label_start_date'].strftime('%Y%m%d')
    else:
        label_start_date = '20210218'

    fc_dt = configs['fc_dt'].strftime('%Y%m%d')
    fc_dt_1 = (configs['fc_dt'] + timedelta(days=-1)).strftime('%Y%m%d')

    actual_sql = """
        select base.*,
           sh.sku_name,
           sh.cat1_id,
           sh.cat2_id,
           sh.brand_id
      from (
            select *
              from mart_caterb2b_forecast.app_caterb2b_forecast_sale_3p_input
             where dt>='{label_start_date}' and dt<'{fc_dt}' and is_train=1

            union 

            select a.* from (
                    ---fc_dt 之后的预测特征
                    select *
                      from mart_caterb2b_forecast.app_caterb2b_forecast_sale_3p_input
                     where dt='{fc_dt}' and is_train=0
                    )a
                    join (
                     ---fc_dt 前一天的的需要预测的sku
                    select distinct bu_id,wh_id,sku_id
                      from mart_caterb2b_forecast.app_caterb2b_forecast_sale_3p_input
                      where dt='{fc_dt_1}' and is_train=1
                    )b
            on a.bu_id = b.bu_id and a.wh_id = b.wh_id and a.sku_id = b.sku_id

           ) base
      JOIN mart_caterb2b.dim_caterb2b_sku_his sh
        ON base.sku_id = sh.sku_id and base.dt = sh.dt
    """.format(label_start_date=label_start_date, fc_dt=fc_dt, fc_dt_1=fc_dt_1)

    schema_cols = spark.sql(actual_sql).schema.names
    print('checking  using table is {},schema_cols is {}'.format(actual_sql, schema_cols))
    raw_data = spark.sql(actual_sql)
    return raw_data


def extract_data(spark, configs):
    '''
    laoding all data for train and predict
    '''
    if 'label_start_date' in configs:
        label_start_date = configs['label_start_date'].strftime('%Y%m%d')
    else:
        label_start_date = '20210218'

    fc_dt = configs['fc_dt'].strftime('%Y%m%d')
    fc_dt_1 = (configs['fc_dt'] + timedelta(days=-1)).strftime('%Y%m%d')

    base_sql = """
             select base.*,
                   sh.cat2_id,
                   sh.brand_id
              from (
                    select sales.*
                      from (
                            select *
                              from mart_caterb2b_forecast.app_sale_bu_wh_sku_3p_day_input
                             where dt>='{label_start_date}'
                               and dt<'{fc_dt}'
                           ) sales
                      join -- 曾经有过销售 或者 上过架的
                           (
                            select distinct bu_id,
                                   wh_id,
                                   sku_id
                              from mart_caterb2b_forecast.app_sale_bu_wh_sku_3p_day_input
                             where dt>='{label_start_date}'
                               and dt<'{fc_dt}'
                               and (arranged_cnt>0 or is_on_shelf>0)
                           ) items
                        on sales.bu_id = items.bu_id
                       and sales.wh_id = items.wh_id
                       and sales.sku_id = items.sku_id
                   )base
              JOIN mart_caterb2b.dim_caterb2b_sku_his sh
                ON base.sku_id = sh.sku_id
               and base.dt = sh.dt
    """.format(label_start_date=label_start_date, fc_dt=fc_dt)

    actual_sql = """
        select base.*,
               sh.sku_name,
               sh.cat1_id,
               sh.cat2_id,
               sh.brand_id
          from (
                select *
                  from mart_caterb2b_forecast.app_caterb2b_forecast_sale_3p_input
                 where dt>='{label_start_date}' and dt<'{fc_dt}' and is_train=1

                union 

                select a.* from (
                        ---fc_dt 之后的预测特征
                        select *
                          from mart_caterb2b_forecast.app_caterb2b_forecast_sale_3p_input
                         where dt='{fc_dt}' and is_train=0
                        )a
                        join (
                        ---fc_dt 前一天的的需要预测的sku
                        select distinct bu_id,wh_id,sku_id
                          from mart_caterb2b_forecast.app_caterb2b_forecast_sale_3p_input
                          where dt='{fc_dt_1}' and is_train=1
                        )b
                on a.bu_id = b.bu_id and a.wh_id = b.wh_id and a.sku_id = b.sku_id

               ) base
          JOIN mart_caterb2b.dim_caterb2b_sku_his sh
            ON base.sku_id = sh.sku_id and base.dt = sh.dt
    """.format(label_start_date=label_start_date, fc_dt=fc_dt, fc_dt_1=fc_dt_1)

    base_data = spark.sql(base_sql)
    # raw_data = spark.sql(actual_sql)

    return base_data, None


def save_to_hive(df,
                 forecast_date,
                 output_path="viewfs://hadoop-meituan/zw02nn45/warehouse/mart_caterb2b_forecast_test.db/{path_keyword}/fc_dt={partition_dt}/model_name={partition_model_name}",
                 path_keyword='app_sale_3p_fc_basic_day_output',
                 model_id=10003,
                 model_name='pop_longtail_baseline'):
    sum_win = Window.partitionBy('bu_id', 'wh_id', 'sku_id')
    agg_df = df.groupby('bu_id', 'wh_id', 'sku_id').pivot('fc_dt').agg(F.first('fcst_rst').alias('fcDt'))
    print("agg_df columns: {}".format(agg_df.columns))
    result_df = agg_df.withColumn('today_fc_cnt', agg_df[forecast_date]).withColumn(
        'daily_fc_cnt', F.regexp_replace(
            F.to_json(F.struct([F.when(F.lit(x).like('20%'), agg_df[x]).alias(x) for x in agg_df.columns])), 'fcDt', '')
    ).select(F.col('bu_id'), F.col('wh_id'), F.col('sku_id'), F.lit(model_id).alias('model_id'),
             F.col('today_fc_cnt'), F.col('daily_fc_cnt'))
    hive_table_path = output_path.format(
        path_keyword=path_keyword, partition_dt=forecast_date,partition_model_name=model_name)
    print(hive_table_path)
    result_df.repartition(10).write.mode('overwrite').orc(hive_table_path)


###########################################date utils###########################################################
def get_date(date_str, day_delta=0, date_format=date_format, output_date_format=date_format):
    return datetime.strftime(datetime.strptime(date_str, date_format) + timedelta(days=day_delta), output_date_format)


def get_dates_gap(first_date_str, second_date_str, date_format=date_format):
    return (datetime.strptime(second_date_str, date_format) - datetime.strptime(first_date_str, date_format)).days


def date_range(start, days, step=1, format="%Y%m%d"):
    strptime, strftime = datetime.strptime, datetime.strftime
    return [strftime(strptime(start, format) + timedelta(i), format) for i in range(0, days, step)]


###########################################date utils###########################################################

def get_sku_router(_rdd, configs):
    """
    sku 路由逻辑
    :param _rdd: spark rdd数据
    :param configs: 配置参数字典对象
    :return:
    """

    import pandas as pd
    sku_data = pd.DataFrame(_rdd[1], columns=configs['cnt_data_columns']).sort_values('dt')
    unique_id = _rdd[0]
    checking_len = configs['sku_router_rules']['checking_len']
    forecast_date = configs['fc_dt'].strftime('%Y%m%d')

    # using checking_len
    sku_data = sku_data[sku_data.dt < forecast_date].tail(checking_len)  # 用评测之前的日志做达标才合适
    sale = sku_data[configs['label']]
    return_list = []

    # check_days	count_sale	fc_dt	mean_sale	router_type	unique_id	sku_id
    # 913	7	4	20210318	11.2	very_long_tail	11002454_136_50780122	50780122

    # case no_sell   # 可售没有销量 or 不可售（肯定没有销量）
    if sku_data.shape[0] >= checking_len:  # 有checking_len天的数据,不算是新品了
        mean_sale = sale.mean()
        count_sale = sale[sale > 0].count()
        if mean_sale <= 0 or count_sale <= 0:  # case no_sell
            return_list.append([unique_id, checking_len, mean_sale, count_sale, 'no_sell', forecast_date])
        elif mean_sale <= 0.5 or count_sale <= 1:  # case_very_long_tail
            return_list.append([unique_id, checking_len, mean_sale, count_sale, 'very_long_tail', forecast_date])
        elif mean_sale <= 10 or count_sale <= 3:  # case_long_tail
            return_list.append([unique_id, checking_len, mean_sale, count_sale, 'long_tail', forecast_date])
        else:  # case normal
            return_list.append([unique_id, checking_len, mean_sale, count_sale, 'normal', forecast_date])
    else:  # 新上品#
        mean_sale = sale.mean()
        count_sale = sale[sale > 0].count()
        if mean_sale <= 0 or count_sale <= 0:  # case no_sell
            return_list.append([unique_id, checking_len, mean_sale, count_sale, 'no_sell', forecast_date])
        else:  # using wma for 新品预测
            return_list.append([unique_id, checking_len, mean_sale, count_sale, 'very_long_tail', forecast_date])

    router_data = pd.DataFrame(return_list,
                               columns=['unique_id', 'check_days', 'mean_sale', 'count_sale', 'router_type', 'fc_dt'])
    final_list = []
    for val in router_data.values:
        final_list.append(dict(zip(router_data.keys(), val)))

    return final_list


def harmonious_mean(ll):
    '''
    对正值计算调和平均数，乘以 正值/总长度
    调和平均：
    受极端值影响比较大，尤其是极小值
    任何一个数不能是0
    :param ll:
    :return:
    '''
    zero_count = 0
    non_zero_count = 0
    weight = 0.0
    for l in ll:
        if l > 0.0:
            weight += 1.0 / l
            non_zero_count = non_zero_count + 1
        else:
            zero_count = zero_count + 1
    adj = (non_zero_count / weight if weight > 0.0 else 0.0)
    return ((adj * non_zero_count) / float(len(ll)) if len(ll) > 0.0 else 0.0)


def seqOp(x, y):
    """seqFunc"""
    x.append(y)
    return x


def combOp(x, y):
    """combFunc"""
    return x + y
