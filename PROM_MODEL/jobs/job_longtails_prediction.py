#!/bin/usr/env python3.6
# -*- coding: utf-8 -*-
# ---------------------------------
# ProjectName: data-forecast-spark
# Description: 销量预测--长尾商品预测
# Author: songzhen07
# CreateTime: 2021-01-15 13:18
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
        app_name='sale_forecast_longtail_sku_prediction')

    # log that main spark job is starting
    log.warn('longtail_sku_prediction is up-and-running')

    # execute spark pipeline
    final_config, data = extract_data(spark, configs)
    forecast_result = train_and_pre(spark, log, data, final_config)
    result_df = transform_data(forecast_result)
    write_data(result_df, final_config)

    # log the success and terminate Spark application
    log.warn('longtail_sku_prediction is finished')
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
    from pyspark.sql.types import DateType
    import pyspark.sql.functions as F

    # 日期参数设定
    current_date = datetime.now().strftime("%Y%m%d")
    if 'fc_dt' in configs:
        current_date = configs['fc_dt']
    ## 预测日期
    forecast_date = (pd.to_datetime(current_date) + timedelta(configs['fc_date_delta'])).strftime('%Y%m%d')
    ## 训练开始/结束日期
    train_start_date = (pd.to_datetime(forecast_date) + timedelta(configs['train_duration'])).strftime('%Y%m%d')
    train_end_date = (pd.to_datetime(forecast_date) - timedelta(1)).strftime('%Y%m%d')
    ## 测试结束日期
    forecast_end_date = (pd.to_datetime(forecast_date) + timedelta(configs['test_duration'])).strftime('%Y%m%d')
    ## 预测数据日期范围
    pred_dates = pd.date_range(forecast_date, forecast_end_date)
    ## 训练数据日期范围
    train_dates = pd.date_range(train_start_date, train_end_date)

    inquire_sql = '''
        select o.dt as date,
               o.bu_id,
               o.wh_id,
               o.sku_id,
               o.arranged_cnt,
               o.able_sell
          from mart_caterb2b_forecast.app_caterb2b_forecast_input_sales_dt_wh_sku o
          join mart_caterb2b.dim_caterb2b_sku s
            on s.sku_id = o.sku_id
          join (
                select bu_id,
                       wh_id,
                       sku_id
                  from mart_caterb2b_forecast.app_caterb2b_forecast_sale_input_v3
                 where is_train = 0
                   and dt = {t0}
                   and cat1_id in ({cats})
               ) v
            on v.bu_id = o.bu_id
           and v.wh_id = o.wh_id
           and v.sku_id = o.sku_id
         where o.on_shelf = 1
           and s.cat1_id in ({cats})
           and o.dt BETWEEN {t1} and {t2}
    '''
    # 得到原始数据集
    raw_data = spark.sql(inquire_sql.format(t0=forecast_date, t1=train_start_date,
                                            t2=train_end_date, cats=configs['cat1_id']))

    ## 数据筛选
    time_func = F.udf(lambda x: datetime.strptime(x, '%Y%m%d'), DateType())
    cnt_data = raw_data.withColumn('dt', time_func(F.col('date')))\
        .filter(~((F.col('able_sell') == 0) & (F.col('arranged_cnt') == 0)))


    ## 更新配置参数字典
    append_conf = {"fc_dt": forecast_date,
                   "bd_col_names": cnt_data.schema.names,
                   "pred_dates": pred_dates,
                   "train_dates": train_dates
                   }
    configs.update(append_conf)

    return configs, cnt_data


def seqOp(x, y):
    """seqFunc"""
    x.append(y)
    return x



def combOp(x, y):
    """combFunc"""
    return x+y


#############################################
# train and pre
#############################################
def train_and_pre(spark, log, cnt_data, configs):
    """
    模型训练&预测
    :param:
        spark: spark对象
        log：日志打印对象
        cnt_data: 输入spark df数据
        configs： 配置参数字典对象

    :return：
        final_df: 训练+测试特征数据
    """

    from pyspark.sql import Window
    import pyspark.sql.functions as F

    log.info("Start longtail sku judgment logic...")
    # 长尾SKU判定
    longtails_rdd = cnt_data.rdd.map(lambda row: ((row['bu_id'], row['wh_id'], row['sku_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=1000, keyfunc=lambda x: x) \
        .flatMap(lambda item: get_longtail_data(item, configs))
    df_longtail = spark.createDataFrame(longtails_rdd)

    # 长尾品销量数据
    log.info("Start to get sales data of longtail sku...")
    cnt_longtail = cnt_data.join(df_longtail.select(F.col('bu_id'), F.col('wh_id'), F.col('sku_id')),
                                 ['bu_id', 'wh_id', 'sku_id'], 'inner')

    log.info("Update configs...")
    append_conf = {'his_sale_days': 28,
                   'lt_col_names': cnt_longtail.schema.names}
    configs.update(append_conf)

    # 长尾商品近28天销量均值
    log.info("Start to calculate historical sales...")
    window = Window.partitionBy(cnt_longtail['bu_id'], cnt_longtail['wh_id'], cnt_longtail['sku_id']) \
        .orderBy(cnt_longtail['dt'].desc())

    avg_result = cnt_longtail.select('*', F.rank().over(window).alias('rank')) \
        .filter(F.col('rank') <= configs['his_sale_days']) \
        .groupby(['bu_id', 'wh_id', 'sku_id']) \
        .agg(F.mean('arranged_cnt').alias('avg_result'))

    # 制造训练矩阵
    log.info("Start train data matrix manufacturing...")
    train_rdd = cnt_longtail.rdd.map(lambda row: ((row['bu_id'], row['wh_id'], row['sku_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=1000, keyfunc=lambda x: x) \
        .flatMap(lambda item: make_train_matrix(item, configs))
    ## 关联均值结果
    df_train = spark.createDataFrame(train_rdd).join(avg_result, ['bu_id', 'wh_id', 'sku_id'], 'inner')

    log.info("Update configs...")
    configs.update({'tr_col_names': df_train.schema.names})

    # 模型训练&预测
    log.info("Start to train and pre...")
    forecast_rdd = df_train.rdd.map(lambda row: ((row['bu_id'], row['wh_id'], row['sku_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=1000, keyfunc=lambda x: x) \
        .flatMap(lambda item: holtwinters_pre(item, configs))

    forecast_result = spark.createDataFrame(forecast_rdd)

    return forecast_result



def get_longtail_data(_rdd, configs):
    """
    长尾判定逻辑
    :param _rdd: spark rdd数据
    :param configs: 配置参数字典对象
    :return: 长尾SKU
    """

    import pandas as pd

    sku_data = pd.DataFrame(_rdd[1], columns=configs['bd_col_names']).sort_values('dt')
    index = list(_rdd[0])
    longtails = []
    result_list = []

    ## 以7日为滑动窗口生成销量序列，均值<7判定为极端长尾
    sale = sku_data['arranged_cnt']
    if sale.count() >= 7:
        roll_sale = sale.rolling(7, min_periods=7).sum()
        avg_sale = roll_sale.mean()
        if avg_sale < 7:
            longtails.append([index[0], index[1], index[2], 7, avg_sale, configs['fc_dt']])
    else:
        ## 不足7日的数据根据实际天数判定
        if sale.sum() < sale.count():
            longtails.append([index[0], index[1], index[2], sale.count(), sale.sum(), configs['fc_dt']])

    longtail_data = pd.DataFrame(longtails, columns=['bu_id', 'wh_id', 'sku_id', 'judge_days', 'sale_avg', 'fc_dt'])
    for val in longtail_data.values:
        result_list.append(dict(zip(longtail_data.keys(), val)))

    return result_list


def make_train_matrix(_rdd, configs):
    """
    训练数据矩阵制造
    :param _rdd: spark rdd数据
    :param configs: 配置参数字典对象
    :return: 训练数据矩阵
    """

    import numpy as np
    import pandas as pd

    sku_data = pd.DataFrame(_rdd[1], columns=configs['lt_col_names']).sort_values('dt')
    index = list(_rdd[0])
    result_list = []

    long = 720 ## 设定序列长度
    sale = sku_data['arranged_cnt'].tolist()

    if len(sale) < long:
        nan_list = [np.nan] * (long - pd.Series(sale).count())
        sale = nan_list + sale
    _train = pd.DataFrame({'bu_id': [index[0]] * long,
                           'wh_id': [index[1]] * long,
                           'sku_id': [index[2]] * long,
                           'cnt_val': sale,
                           'date': configs['train_dates'].strftime('%Y%m%d')
                           })
    for val in _train.values:
        result_list.append(dict(zip(_train.keys(), val)))
    return result_list



def holtwinters_pre(_rdd, configs):
    """
    训练&预测
    :param _rdd: spark rdd数据
    :param configs: 配置参数字典对象
    :return: 预测结果
    """

    import pandas as pd
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    values = pd.DataFrame(_rdd[1], columns=configs['tr_col_names']).sort_values('date').dropna()
    index = list(_rdd[0])
    result_list = []

    # 数据为空
    if values.count()[0] == 0:
        return result_list

    avg_val = values.avg_result.iloc[0]
    data = values.cnt_val
    periods = 4

    try:
        mod = ExponentialSmoothing(data,
                                   trend='add',
                                   seasonal='add',
                                   #damped=True,
                                   seasonal_periods=periods)
        hlt_model = mod.fit()
        preds = hlt_model.forecast(20)
        preds[preds < 0] = 0
        pred_uc = preds.values[0]
    except:
        pred_uc = data.iloc[-1]
        preds = pd.Series([pred_uc] * 20)
    finally:
        degraded_array = pd.Series([random.uniform(0.001, 0.01) for i in range(20)])
        t0_except = preds[1:] + 0.001
        day_ratio = t0_except / t0_except.sum()
        avg_day_pred = (avg_val * 20 - pred_uc) * day_ratio
        pred_rst = pd.concat([pd.Series(pred_uc), avg_day_pred])
        pred_rst = np.where(pred_rst < 0, degraded_array,  pred_rst)
        hw_result = pd.DataFrame({'bu_id': [index[0]] * 20,
                                  'wh_id': [index[1]] * 20,
                                  'sku_id': [index[2]] * 20,
                                  'prediction': pred_rst,
                                  'date': configs['pred_dates'].strftime('%Y%m%d')
                                  })
    for val in hw_result.values:
        result_list.append(dict(zip(hw_result.keys(), val)))

    return result_list


#############################################
# transform data
#############################################
def transform_data(forecast_result):
    """
    预测结果格式化
    :param forecast_result: 预测值原始形式
    :param configs: 配置参数字典对象
    :return: json格式预测结果
    """

    from pyspark.sql import Window
    import pyspark.sql.functions as F

    sum_win = Window.partitionBy('bu_id', 'wh_id', 'sku_id')

    agg_df = forecast_result.withColumn('total_amt_fc_wk', F.sum('prediction').over(sum_win)) \
        .groupby('bu_id', 'wh_id', 'sku_id', 'total_amt_fc_wk') \
        .pivot('date') \
        .agg(F.first('prediction').alias('pred'))

    result_df = agg_df.withColumn('daily_amt_fc_wk', F.regexp_replace(F.to_json(F.struct([F.when(F.lit(x).like('20%'), agg_df[x]).alias(x) for x in agg_df.columns])), 'pred', '')) \
        .select(F.col('bu_id'), F.col('wh_id'), F.col('sku_id'), F.lit(0).alias('source'), F.lit(5001).alias('model'), F.col('total_amt_fc_wk'), F.col('daily_amt_fc_wk'))

    return result_df



#############################################
# output data
#############################################
def write_data(df, configs):
    """Collect data locally and write to CSV.

    :param: rdd to print.
    :return: None
    """
    df.repartition(10).write.mode('overwrite') \
        .orc(configs['output_path'].format(str(configs['fc_dt'])))
    return None


#############################################
# entry point for PySpark application
#############################################
if __name__ == '__main__':
    main()








