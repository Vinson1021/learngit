# coding: utf-8

import getopt
import sys
from datetime import datetime, timedelta
import time
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import Window

date_format = '%Y-%m-%d'
date_format_compat = '%Y%m%d'
time_window = 7 * 4
min_gap_day = 1
feature_cols = 'median_arranged_cnt_7,max_arranged_cnt_7,min_arranged_cnt_7,avg_arranged_cnt_7,median_arranged_cnt_14,max_arranged_cnt_14,min_arranged_cnt_14,avg_arranged_cnt_14,median_arranged_cnt_28,max_arranged_cnt_28,min_arranged_cnt_28,avg_arranged_cnt_28,is_work_day,is_weekend,is_holiday,date_gap,day_abbr_num,new_byrs,old_byrs,fresh_byrs,vege_byrs'.split(',')


def robust_op(ll, op=np.nanmean, default=0.0):
    r = op(ll)
    return default if np.isnan(r) else r


def compute_mapd(reals, predicts):
    ape_list = np.where(reals == 0.0, np.where(
        predicts < 1, 0, 1), np.abs(reals - predicts))
    total_reals = robust_op(reals, op=np.nansum)
    total_abs_diff = robust_op(np.abs(reals - predicts), op=np.nansum)
    wmape = np.sum(ape_list * reals) / \
        total_reals if total_reals != 0.0 else total_abs_diff
    return wmape


def get_date(date_str, day_delta=0, date_format=date_format, output_date_format=date_format):
    return datetime.strftime(datetime.strptime(date_str, date_format) + timedelta(days=day_delta), output_date_format)


def get_dates_gap(first_date_str, second_date_str, date_format=date_format):
    return (datetime.strptime(second_date_str, date_format) - datetime.strptime(first_date_str, date_format)).days


def date_range(start, days, step=1, format="%Y%m%d"):
    strptime, strftime = datetime.strptime, datetime.strftime
    return [strftime(strptime(start, format) + timedelta(i), format) for i in range(0, days, step)]


def get_argv(argv):
    argv = argv[1:]
    target_cat1_id = ''
    try:
        opts, args = getopt.getopt(argv, "c:", ["cat1_id="])
    except getopt.GetoptError:
        print('test.py -c <cat1_id>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-c", "--cat1_id"):
            target_cat1_id = arg
    return target_cat1_id


def pd_print(tdf):
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    print(tdf)
    pd.options.display.max_columns = 20
    pd.options.display.max_rows = 60


def harmonious_mean(ll):
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
    return (adj * non_zero_count) / float(len(ll)) if len(ll) > 0.0 else 0.0


def save_to_hive(df, forecast_date, path_keyword='app_caterb2b_forecast_sale_data_augmentation_output_da_tmp'):
    sum_win = Window.partitionBy('wh_id', 'sku_id')
    agg_df = df.withColumn('total_amt_fc_wk', F.sum('fcst_rst').over(sum_win)).groupby('wh_id', 'sku_id', 'total_amt_fc_wk').pivot('fc_dt').agg(F.first('fcst_rst').alias('fcDt'))
    result_df = agg_df.withColumn('daily_amt_fc_wk', F.regexp_replace(F.to_json(F.struct([F.when(F.lit(x).like('20%'), agg_df[x]).alias(x) for x in agg_df.columns])), 'fcDt', '')).select(F.col('wh_id'), F.col('sku_id'), F.lit(0).alias('source'), F.lit(5).alias('model'), F.col('total_amt_fc_wk'), F.col('daily_amt_fc_wk'))

    hive_table_path = "viewfs://hadoop-meituan/zw02nn45/warehouse/mart_caterb2b_forecast.db/{path_keyword}/fc_dt={partition_dt}/".format(path_keyword=path_keyword, partition_dt=forecast_date)
    print(hive_table_path)
    result_df.repartition(10).write.mode('overwrite').csv(hive_table_path, sep='\001', quote='', escape='', escapeQuotes=False)


def log_print(bodyStr, level='INFO'):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " " + level + " " + bodyStr)


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
