#!/bin/usr/env python3.6
# -*- coding: utf-8 -*-
# ---------------------------------
# ProjectName: data-forecast-spark
# Description: 3P 离线数据输出整合模型 10003
# Author: lijingjie
# file:job_pop_longtail_output.py
# CreateTime: 2021-04-14 22:35
# ---------------------------------
# 设置随机性
import numpy as np
import random
import os
seed=44
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

import os, sys
import json
import datetime
import pandas as pd
import numpy as np
from itertools import chain
from pyspark.sql.types import *
from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import *
from functools import reduce
from pyspark.sql import DataFrame
from pyspark import SparkFiles

import numpy as np
from dependencies.platform.spark import start_spark
from dependencies.algorithm.pop.preprocess import extract_data, seqOp, combOp, save_to_hive

#### -----> 模型预测输出    支持多步预测
# 'bu_id,wh_id,sku_id,fc_dt,inter_fcst_rst'
#### -----> 最终输出结果表，需要调用函数 save_to_hive
schema = StructType([StructField("bu_id", LongType(), True), StructField("wh_id", LongType(), True),
                     StructField("sku_id", LongType(), True), StructField("fc_dt", StringType(), True),
                     StructField("fcst_rst", DoubleType(), True)])
#############################################
# main function
#############################################

def main():
    """Main script definition.
    :return: None
    """
    # start Spark application and get Spark session, logger and config
    spark, log, configs = start_spark(app_name='job_pop_longtail_output')
    # 日期参数设定
    configs['fc_dt'] = pd.to_datetime(configs['fc_dt']) if 'fc_dt' in configs else datetime.datetime.now().date()
    configs['today'] = pd.to_datetime(configs['fc_dt']) + datetime.timedelta(
        days=-1) if 'fc_dt' in configs else datetime.datetime.now().date() + datetime.timedelta(days=-1)
    configs['label_start_date'] = pd.to_datetime(
        configs['label_start_date']) if 'label_start_date' in configs else pd.to_datetime('20210220').date()

    env_is_test = '_test' if 'test' in configs['output_path'] else ''

    # log that main spark job is starting
    log.warn('job_pop_longtail_output is up-and-running')
    print('job_pop_longtail_output is up-and-running ,attention merged_model_names is ordered', configs['fc_dt'], configs['today'],configs['base_model_name'],configs['merged_model_names'])

    ## construct final replace sql
    ### step1 construct condition should be like
    # 'if(new_model_0.today_fc_cnt is null,if(new_model_1.today_fc_cnt is null,base_model.today_fc_cnt,new_model_1.today_fc_cnt),new_model_0.today_fc_cnt) as today_fc_cnt'
    new_model_names_len = len(configs['merged_model_names'])
    if new_model_names_len == 1:
        condition_choose_str = """if(new_model_0.today_fc_cnt is null,base_model.today_fc_cnt,new_model_0.today_fc_cnt) as fcst_rst"""
    else:
        condition_choose = ["""if(new_model_0.today_fc_cnt is null,{},new_model_0.today_fc_cnt) as fcst_rst"""]
        for i in range(1, new_model_names_len):
            if i == new_model_names_len - 1:  ## last one
                condition_choose.append(
                    """if(new_model_""" + str(i) + """.today_fc_cnt is null,base_model.today_fc_cnt,new_model_""" + str(
                        i) + """.today_fc_cnt)""")
            else:
                condition_choose.append(
                    """if(new_model_""" + str(i) + """.today_fc_cnt is null,{},new_model_1.today_fc_cnt)""")

        condition_choose_str = condition_choose[0]
        for i in range(1, len(configs['merged_model_names'])):
            print(i)
            condition_choose_str = condition_choose_str.format(condition_choose[i])
    ### step2 final sql
    overal_sql = """
            select base_model.bu_id,
               base_model.wh_id,
               base_model.sku_id,
               base_model.fc_dt,
               """ + condition_choose_str + """
          from (
                select bu_id,
                       wh_id,
                       sku_id,
                       fc_dt,
                       today_fc_cnt
                  from mart_caterb2b_forecast"""+env_is_test+""".app_sale_3p_fc_basic_day_output
                 where model_name = '"""+configs['base_model_name']+"""' and fc_dt='"""+configs['fc_dt'].strftime('%Y%m%d')+"""'
               )base_model
          left join(
                select bu_id,
                       wh_id,
                       sku_id,
                       fc_dt,
                       today_fc_cnt
                  from mart_caterb2b_forecast"""+env_is_test+""".app_sale_3p_fc_basic_day_output
                 where model_name = '"""+configs['merged_model_names'][0]+"""' and fc_dt='"""+configs['fc_dt'].strftime('%Y%m%d')+"""'
               )new_model_0
          on  base_model.bu_id = new_model_0.bu_id
          and  base_model.wh_id = new_model_0.wh_id
          and  base_model.sku_id = new_model_0.sku_id
          """
    for i in range(1, new_model_names_len):
        overal_sql += """
        left join (
              select bu_id,
                       wh_id,
                       sku_id,
                       fc_dt,
                       today_fc_cnt
                  from mart_caterb2b_forecast"""+env_is_test+""".app_sale_3p_fc_basic_day_output
                 where model_name = '"""+configs['merged_model_names'][i]+"""' and fc_dt='"""+configs['fc_dt'].strftime('%Y%m%d')+"""'
               )new_model_""" + str(i) + """
          on  base_model.bu_id = new_model_""" + str(i) + """.bu_id
          and  base_model.wh_id = new_model_""" + str(i) + """.wh_id
          and  base_model.sku_id = new_model_""" + str(i) + """.sku_id
        """
    print('overall replace sql is ',overal_sql)

    combine_longtail = spark.sql(overal_sql)

    # STEP 3 final output
    ##### 最终输出 'bu_id','wh_id','sku_id','model','total_fc_cnt,daily_fc_cnt'
    print('job_pop_longtail_bht_arima,construct fianl output ', combine_longtail.count(),
          combine_longtail.take(1))
    save_to_hive(combine_longtail, configs['fc_dt'].strftime('%Y%m%d'), model_id=configs['model_id'],
                 output_path=configs['output_path'], model_name=configs['model_name'])

    ## save data
    # log the success and terminate Spark application
    log.warn('job_pop_longtail_output is finished')
    spark.stop()
    return None


#############################################
# entry point for PySpark application
#############################################
if __name__ == '__main__':
    main()
