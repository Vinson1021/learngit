# -*- coding:utf-8 -*-
'''
@author : gaotong04
@time : 2021/7/7 10:18 上午
@输入sql：
@输出表名：
'''

# 基本

import pandas as pd
import random
import datetime
import numpy as np
# spark
from pyspark.sql import Row
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.functions import col, concat_ws, lit
from pyspark.sql.types import StructType, StructField, LongType, StringType,DoubleType
# 框架
from dependencies.platform.spark import start_spark
from dependencies.algorithm.pop.preprocess import extract_data, seqOp, combOp, save_to_hive

#### -----> 模型预测输出    支持多步预测
# 'bu_id,wh_id,sku_id,fc_dt,inter_fcst_rst'
#### -----> 最终输出结果表，需要调用函数 save_to_hive

def main():
    """Main script definition.
    :return: None
    """
    # start Spark application and get Spark session, logger and config
    spark, log, config = start_spark(
        app_name='job_indicator_xgb_spark')

    # log that main spark job is starting
    log.warn('job_indicator_xgb_spark is up-and-running')

    # execute spark pipeline
    data = extract_data(spark, config,log)
    data_transformed = transform_data(data, config,log,spark)
    write_data(data_transformed, config,log,spark)

    # log the success and terminate Spark application
    log.warn('job_demo is finished')
    spark.stop()
    return None

def extract_data(spark, config,log):
    """Load data from Parquet file format.

    :param spark: Spark session object.
    :param config: Config params from json.
    :return: Spark DataFrame.
    """

    # 指定预测日期
    config['fc_dt'] = config['fc_dt'] if 'fc_dt' in config else datetime.datetime.now().date().strftime('%Y%m%d')

    # train_start_date 获取
    config['train_start_dt'] =  config['train_start_dt'] if 'train_start_dt' in config else '20210220'

    config['today'] = (pd.to_datetime(config['fc_dt']) + datetime.timedelta(
        days=-1)).strftime('%Y%m%d') if 'fc_dt' in config else (datetime.datetime.now().date() + datetime.timedelta(days=-1)).strftime('%Y%m%d')

    spark_sql = '''
    select base.*,
          sh.cat2_id,
          sh.brand_id,
          UV.view_uv,
          UV.view_outstock_uv,
          UV.outstock_uv,
          UV.intent_uv,
          UV.order_customer_cnt,
          UV.arranged_price,
          UV.original_price,
          UV.platform_arranged_price,
          UV.platform_original_price

     from (
           select sales.*
             from (
                   select *
                     from mart_caterb2b_forecast.app_sale_bu_wh_sku_3p_day_input
                    where dt>='{start_date}'
                      and dt<'{end_date}'
                  ) sales
             join (
                   select distinct bu_id,
                          wh_id,
                          sku_id
                     from mart_caterb2b_forecast.app_sale_bu_wh_sku_3p_day_input
                    where dt>='{start_date}'
                      and dt<'{end_date}'
                      and (arranged_cnt>0 or is_on_shelf>0)
                  ) items
               on sales.bu_id = items.bu_id
              and sales.wh_id = items.wh_id
              and sales.sku_id = items.sku_id
          )base
     join (
           select bu_id,
                  wh_id,
                  sku_id,
                  avg(arranged_cnt) AS avg_cnt
             from mart_caterb2b_forecast.app_sale_bu_wh_sku_3p_day_input
            where dt>='{start_date}'
              and dt<'{end_date}'
            group by bu_id,
                     wh_id,
                     sku_id
           having avg_cnt>5
          ) filter
       on filter.bu_id = base.bu_id
      and filter.wh_id = base.wh_id
      and filter.sku_id = base.sku_id join(
           select bu_id,
                  sku_id,
                  dt,
                  view_outstock_uv,
                  outstock_uv,
                  intent_uv,
                  order_customer_cnt,
                  arranged_price,
                  original_price,
                  platform_arranged_price,
                  platform_original_price,
                  view_uv
             from mart_caterb2b.total_prod_3p_city_bu_sku_day
            where dt>='{start_date}'
              and dt<'{end_date}'
          ) UV
       on base.bu_id=UV.bu_id
      and base.sku_id=UV.sku_id
      and base.dt=UV.dt
     JOIN mart_caterb2b.dim_caterb2b_sku_his sh
       ON base.sku_id = sh.sku_id
      and base.dt = sh.dt
    '''.format(start_date =config['train_start_dt'] , end_date=config['fc_dt'])
    print(spark_sql)
    data_df = spark.sql(spark_sql).persist()
    # config.update({'forecast_date': forecast_date})
    print('读取数据行数：',data_df.count())

    return data_df

def partitionForecast(data_in_partition, config):
    # 下面的包均是在executor端引入，不要调整到文件首部，保证他们在方法内被导入
    import sys,os
    import pandas as pd
    from dependencies.algorithm.sale_forecast_pop.pop_day_xgb import POPXGBModel

    seed = 2021
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    os.environ['PYTHONPATH'] = './envs/site-packages:' + os.environ['PYTHONPATH']
    sys.path.insert(0, './envs/site-packages')  # 环境依赖
    # sys.path.append('ARCHIVE/notebook/lib/python3.6/site-packages')
    print(sys.path)

    df = pd.DataFrame(data_in_partition, columns=config['columns'])
    # df = df.drop_duplicates()

    print('pandas df 行数',len(df))

    result = None
    if df.empty:
        result = None
    else:
        print('unique sku 个数',df['sku_id'].nunique())
        addrFore = POPXGBModel(df, fc_dt=config['fc_dt'], forecast_step=config['forecast_step'])
        result = addrFore.train_forecast()

    yield result


def transform_data(df, config,log,spark):
    """Transform original dataset.

    :param df: Input DataFrame.
    :param config: Config params from json.
    :return: Transformed DataFrame.
    """

    df = df.withColumn('unique_id', F.concat(F.col('bu_id'), F.lit('_'), F.col('wh_id'), F.lit('_'), F.col('sku_id')))

    ### 动态划分每个executor上的数据量，保证同一sku在同一executor上
    log.info("Start longtail sku judgment logic...")

    ##计算均值
    df.registerTempTable("cnt_data_hist_table")
    mean_saled_days = """
                        select dt,
                              unique_id,
                              arranged_cnt,
                              percentile(arranged_cnt, 0.5) over (partition by unique_id order by dt rows between 20 preceding and 0 preceding) as median_value,
                              avg(arranged_cnt) over (partition by unique_id order by dt rows between 20 preceding and 0 preceding) as mean_value,
                              count(arranged_cnt) over (partition by unique_id order by dt rows between 20 preceding and 0 preceding)  as count_value,
                              sum(arranged_cnt) over (partition by unique_id order by dt rows between 6 preceding and 0 preceding)  as sum_7_value
                         from cnt_data_hist_table
                        where dt >= '20210218'"""

    mean_saled_days_df = spark.sql(mean_saled_days)
    mean_saled_days_df.registerTempTable("mean_saled_days_df_table")

    # 按照时间顺序最后一个dt的mean值来划分数据
    cnt_data_with_mean_df = spark.sql("""
                        select a.*,b.mean_value from (select * 
                        from cnt_data_hist_table) a
                        join (
                        select e.unique_id,e.mean_value from 
                            (select unique_id,mean_value,dt
                            from mean_saled_days_df_table) e
                            join
                            (select max(dt) as dt,unique_id
                            from mean_saled_days_df_table
                            group by unique_id
                            ) f
                            on e.unique_id = f.unique_id
                            and e.dt = f.dt
                           )   b      
                        on a.unique_id = b.unique_id
                        """)

    #按均值顺序打group_label，相似均值的分在同一分区训练一个xgb

    #按均值排序iD
    unique_ids = cnt_data_with_mean_df.select('unique_id', 'mean_value').distinct().orderBy('mean_value').select('unique_id').rdd.map(
        lambda x: x.unique_id).collect()
    #按ID顺序分配编号
    sku_ids = [i for i in range(len(unique_ids))]
    total_sku = len(unique_ids)
    unique_ids_dict = dict(zip(unique_ids, sku_ids))

    @udf(returnType=StringType())
    def get_unique_id(s):
        return unique_ids_dict[s]

    df = df.withColumn('unique_id_int', get_unique_id('unique_id'))
    fitting_size = config['fitting_size']  # 一个executor上放多少个sku
    df = df.drop('unique_id')
    df = df.drop('mean_value')


    # 编号在同一fitting_size内的分到同一分区
    dynamic_str = ' '.join(["""when unique_id_int <""" + str(i * fitting_size) \
                            + """ then 'group_""" + str(i) + """'"""  \
                            for i in range(1, total_sku // fitting_size + 2)])
    dynamic_str = """ case """ + dynamic_str + """end as group_label"""


    df.registerTempTable("df_table")
    df_to_fit = spark.sql("""select *,""" + dynamic_str + """ from df_table""")

    df_to_fit = df_to_fit.drop('unique_id_int')

    # print(df_to_fit.select('unique_id','group_label').distinct().rdd.collect())
    # df = df.drop('unique_id')
    df_to_fit = df_to_fit.repartition(200, df_to_fit.group_label)

    print('处理数据行数：', df_to_fit.count())
    config['columns'] = list(df_to_fit.columns)
    #same up !!!!!!
    #
    result = df_to_fit.rdd.mapPartitions(lambda par: partitionForecast(par,config)) \
                          .filter(lambda x: x is not None)\
                          .persist()
                          # 这里lambda x ， x应该是一个dataframe，因为partitionForecast返回一个df
    # TODO:确认是否需要过滤数据？如长尾品等

    return result

def write_data(result_df_list, config,log,spark):
    """Collect data locally and write to CSV.
    :param df: DataFrame to print.
    :return: None
    """

    log.warn('writing data to hive,spark action is done in this section, it will take a while...')
    spark.sql("set hive.exec.dynamic.partition=true")
    spark.sql("set hive.exec.dynamic.partition.mode=nostrick")

    result_df = result_df_list.reduce(lambda x, y: pd.concat([x, y])) #result_df 是pandas

    result_df = spark.createDataFrame(result_df, config['out_put_dataframe_cols'])
    print(result_df.show(5))
    #为了与输出表对其，还要再创建新的2列
    result_df = result_df.withColumn('model_id', lit(config['model_id']))
    # result_df = result_df.withColumn('model_name', lit('sales_3p_sparkxgb'))

    result_df.persist()
    print(result_df.show(5))

    final_df = result_df.select(F.col('bu_id'), F.col('wh_id'), F.col('sku_id'), F.lit(config['model_id']).alias('model_id'),
             F.col('today_fc_cnt'), F.col('daily_fc_cnt'))

    print(final_df.show(5))

    hive_table_path = config['output_path'].format(path_keyword=config['out_put_tabel_name'], partition_dt=config['fc_dt'], partition_model_name=config['model_name'])
    print(hive_table_path)
    final_df.repartition(10).write.mode('overwrite').orc(hive_table_path)



    print('write finish1')

if __name__ == '__main__':
    main()
