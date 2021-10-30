"""
job_pricing_strategy_feature.py
~~~~~~~~~~

量价模型-特征数据生产任务

"""

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
        app_name='pricing_strategy_feature_engineering')

    # log that main spark job is starting
    log.warn('pricing strategy is up-and-running')

    # execute spark pipeline
    final_config, data = extract_data(spark, configs)
    t0_df, t1_df = transform_data(spark, log, data, final_config)
    write_data(t0_df, t1_df, final_config)

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
    ## 训练开始结束日期
    train_start_date = (pd.to_datetime(forecast_date) + timedelta(configs['train_duration'])).strftime('%Y%m%d')
    train_end_date = (pd.to_datetime(forecast_date) - timedelta(1)).strftime('%Y%m%d')
    ## 测试结束日期
    forecast_end_date = (pd.to_datetime(forecast_date) + timedelta(configs['test_duration'])).strftime('%Y%m%d')

    inquire_sql = '''
        select sales_grid_id,sku_id,cat1_name,cat2_name,cnt_band,brand_name,tax_rate,price,w_price,seq_num,seq_price,redu_num,csu_redu_num,pro_num,pro_price,day_abbr,
        is_work_day,is_weekend,is_holiday,festival_name,western_festival_name,weather,weather_b,avg_t,avg_t_b,new_byrs,old_byrs,poultry_byrs,vege_fruit_byrs,
        egg_aquatic_byrs,standard_byrs,vege_byrs,meet_byrs,frozen_meat_byrs,egg_byrs,is_outstock,arranged_cnt,dt,is_train 
         from {} 
        where dt between {} and {}
        and is_train = 1
        union all 
        select sales_grid_id,sku_id,cat1_name,cat2_name,cnt_band,brand_name,tax_rate,price,w_price,seq_num,seq_price,redu_num,csu_redu_num,pro_num,pro_price,day_abbr,
        is_work_day,is_weekend,is_holiday,festival_name,western_festival_name,weather,weather_b,avg_t,avg_t_b,new_byrs,old_byrs,poultry_byrs,vege_fruit_byrs,
        egg_aquatic_byrs,standard_byrs,vege_byrs,meet_byrs,frozen_meat_byrs,egg_byrs,is_outstock,arranged_cnt,dt,is_train  
         from {} 
        where dt between {} and {}
        and is_train = 0
    '''
    # 得到原始数据集
    raw_data = spark.sql(inquire_sql.format(configs['pricing_stragety_input_table'],
                                            train_start_date, train_end_date,
                                            configs['pricing_stragety_input_table'],
                                            forecast_date, forecast_end_date)).withColumnRenamed('dt', 'date')

    bd_col_names = raw_data.schema.names

    ## 更新配置参数字典
    append_conf = {"current_date": current_date,
                   "forecast_date":forecast_date,
                   "train_start_date":train_start_date,
                   "train_end_date": train_end_date,
                   "forecast_end_date":forecast_end_date,
                   "bd_col_names": bd_col_names
                   }
    configs.update(append_conf)

    return configs, raw_data

#############################################
# data processing 
#############################################
def transform_data(spark, log, _df, configs):
    """生产特征数据.

    :param:
        spark: spark对象
        log：日志打印对象
        _df: 输入spark df数据
        configs： 配置参数字典对象

    :return：
        t0_df: t0 训练+测试数据集
        test_df_t1： t1 测试数据集
    """
    # 预处理数据
    log.info("Start to preprocess...")
    grid_rdd = _df.rdd.map(lambda row:((row['sales_grid_id']),list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=100, keyfunc=lambda x:x) \
        .flatMap(lambda item: data_preprocess(item, configs))
    grid_df = spark.createDataFrame(grid_rdd)

    log.info("Update configs by grid schema...")
    grid_col_names = grid_df.schema.names
    configs.update({"grid_col_names": grid_col_names})

    # 滑动异常值过滤
    log.info("Start filter outliers by windows operation...")
    cleaned_rdd = grid_df.rdd.map(lambda row: ((row['sales_grid_id'], row['sku_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=1000, keyfunc=lambda x: x) \
        .flatMap(lambda item: sku_to_drop(item, configs))
    cleaned_df = spark.createDataFrame(cleaned_rdd)

    # 售卖区域特征制造
    feature_rdd = cleaned_df.rdd.map(lambda row:((row['sales_grid_id']), list(row))) \
        .aggregateByKey(list(), seqOp, combOp) \
        .sortByKey(numPartitions=100, keyfunc=lambda x:x) \
        .flatMap(lambda item: features_create(item, configs))
    feature_df = spark.createDataFrame(feature_rdd)

    train_df = feature_df.where("is_train = 1 or is_train = 3")
    test_df_t0 = feature_df.where("is_train = 0 and date = {}".format(configs['forecast_date']))
    test_df_t1 = feature_df.where("is_train = 0 and date = {}".format(configs['forecast_end_date']))

    t0_df = train_df.union(test_df_t0)

    return t0_df, test_df_t1



def seqOp(x, y):
    '''seqFunc'''
    x.append(y)
    return x



def combOp(x, y):
    '''combFunc'''
    return x+y



def data_preprocess(grid_rdd, configs):
    """
    数据预处理
    """
    import pandas as pd
    import numpy as np
    from dependencies.algorithm.pricingStrategy.pricing_strategy_preprocess_util_v2 import DataUtil

# 转换为pandas DataFrame
    df_raw = pd.DataFrame(grid_rdd[1], columns=configs['bd_col_names']) \
        .replace(np.nan, np.nan)
    df_raw['dt'] = pd.to_datetime(df_raw['date'])
    sales_grid_id = df_raw.sales_grid_id.unique()[0]

    print('正在数据预处理, 详情: ')
    print('--------------------------------')
    print('售卖区域Id: {}, 数据条目: {}\n'.format(sales_grid_id, len(df_raw)))

    df_train_raw = df_raw[df_raw.is_train == 1].sort_values('date').reset_index(drop=True)
    df_pred_raw = df_raw[df_raw.is_train == 0].sort_values('date').reset_index(drop=True)

    # 数据预处理
    ##################################
    ## 剔除缺货日期的数据
    # df_train_raw = df_train_raw[~(df_train_raw.is_outstock == 1)]
    ## 过滤售价为0的数据
    df_train_raw = df_train_raw[df_train_raw['price'] != 0]
    df_pred_raw = df_pred_raw[df_pred_raw['price'] != 0]
    ## 过滤测试数据不足40条的sku
    sku_size = df_pred_raw.groupby('sku_id').size()
    less_20d_skus = sku_size[sku_size < 40].index
    df_pred_raw = df_pred_raw[~df_pred_raw.sku_id.isin(less_20d_skus)]

    try:
        ## 调用预处理模块
        pro_obj = DataUtil(configs['forecast_date'])
        df_train_total = pro_obj.preprocess(df_train_raw, is_train=True)
        ## 过滤近30天未售卖的sku
        sku_train_list = pro_obj.sku_to_train(df_train_raw)
        ## 保留全部数据, 更改需要剔除数据的‘is_train’标签
        df_train_total.loc[~df_train_total['sku_id'].isin(sku_train_list), 'is_train'] = 1024
        df_test = pro_obj.preprocess(df_pred_raw[df_pred_raw['sku_id'].isin(sku_train_list)], is_train=False)
    except ValueError:
        print('{} 数据预处理结果为空！'.format(sales_grid_id))
        return []
    print('{} 数据预处理已完成！'.format(sales_grid_id))

    df_all = pd.concat([df_train_total, df_test])

    result_list = []
    for val in df_all.values:
        result_list.append(dict(zip(df_all.keys(), val)))

    return result_list


def sku_to_drop(sku_rdd, configs):
    """
    滑动过滤异常值
    """
    import pandas as pd

    df_all = pd.DataFrame(sku_rdd[1], columns=configs['grid_col_names'])
    keys = df_all.keys()
    result_list = []
    droped_index = []

    df_sku = df_all[df_all.is_train == 1]


    # 不对数据量小于7的sku过滤
    if df_sku.count()[0] < 7:
        for val in df_all.values:
            result_list.append(dict(zip(keys, val)))
        return result_list

    # 常规日期数据
    data_normal = df_sku[(df_sku['pro_num'] == 0) &
                         (df_sku['seq_num'] == 0) &
                         (df_sku['csu_redu_num'] == 0)]['arranged_cnt']

    # 促销日期数据
    data_pro = df_sku[(df_sku['pro_num'] != 0) |
                      (df_sku['seq_num'] != 0) |
                      (df_sku['csu_redu_num'] != 0)]['arranged_cnt']

    # 滑动窗口获取异常值索引
    if data_normal.count() >= 7:
        for i in range(7, data_normal.count()+1):
            window = data_normal.iloc[i-7:i]
            avg = window.mean()
            droped_index.extend(window[(window < avg / 5.) |
                                       (window > avg * 2.5)].index)
    else:
        avg = data_normal.mean()
        droped_index.extend(data_normal[(data_normal < avg / 5.) |
                                        (data_normal > avg * 2.5)].index)

    if data_pro.count() >= 7:
        for j in range(7, data_pro.count()+1):
            window = data_pro.iloc[j-7:j]
            avg = window.mean()
            droped_index.extend(window[(window < avg / 5.) |
                                       (window > avg * 2.5)].index)
    else:
        avg = data_pro.mean()
        droped_index.extend(data_pro[(data_pro < avg / 5.) |
                                     (data_pro > avg * 2.5)].index)

    # 保留全部数据，更改需要剔除数据的‘is_train’标签
    df_all.loc[df_all.index.isin(droped_index), 'is_train'] = 1024

    for val in df_all.values:
        result_list.append(dict(zip(keys, val)))

    return result_list


def features_create(cleaned_rdd, configs):
    """
    特征制造函数
    """
    import pandas as pd
    from dependencies.algorithm.pricingStrategy.pricing_strategy_feature_util_v4 import FeatureUtil
    from dependencies.algorithm.pricingStrategy.pricing_strategy_price_util_v2 import PriceUtil

    df_all = pd.DataFrame(cleaned_rdd[1], columns=configs['grid_col_names'])
    df_all.fillna(0., inplace=True)
    df_all['dt'] = pd.to_datetime(df_all.date)
    sales_grid_id = df_all.sales_grid_id.unique()[0]
    result_list = []

    df_train = df_all[df_all.is_train == 1]
    df_test = df_all[df_all.is_train == 0]

    # 确保训练集&测试集的sku对应
    common_sku = set(df_train.sku_id.unique()) & set(df_test.sku_id.unique())
    df_train = df_train[df_train.sku_id.isin(common_sku)]
    df_test = df_test[df_test.sku_id.isin(common_sku)]

    if (df_train.count()[0] == 0) or (df_test.count()[0] == 0):
        print("{} 区域近30天未售卖, 预测结果为空!".format(sales_grid_id))
        return result_list

    data = pd.concat([df_train, df_test])
    all_train_data = df_all[df_all.is_train.isin([1, 1024])]

    # dum_list = ['day_abbr', 'festival_name', 'western_festival_name', 'weather', 'weather_b']
    map_list = ['cat1_name', 'day_abbr', 'festival_name', 'western_festival_name', 'weather', 'weather_b', 'year']

    # 日期处理
    data['year'] = pd.to_datetime(data.date).dt.year
    data['month'] = pd.to_datetime(data.date).dt.month
    data['day'] = pd.to_datetime(data.date).dt.day
    all_train_data['year'] = pd.to_datetime(all_train_data.date).dt.year
    all_train_data['month'] = pd.to_datetime(all_train_data.date).dt.month
    all_train_data['day'] = pd.to_datetime(all_train_data.date).dt.day

    print('正在特征制造, 详情: ')
    print('--------------------------------')
    print('售卖区域Id: {}, 数据条目: {}'.format(sales_grid_id, len(data)))

    # 派生特征制造
    try:
        feature_obj = FeatureUtil()
        price_obj = PriceUtil()
        data = feature_obj.week_sale_ratio(data, all_train_data)
        data = feature_obj.cnt_band_func(data)
        data = feature_obj.seasonal_count(data, all_train_data)
        data = feature_obj.mov_arr_avg(data)
        # data = feature_obj.pro_statistic_features(data)
        data = price_obj.price_derived_features(data)
        data = price_obj.get_price_elasticity(data)
        data = price_obj.get_price_ratio_elasticity(data)
        data = price_obj.non_linear_elasticity(data)
        data = price_obj.ed_pred_sale(data)
        data = price_obj.get_price_statistic_features(data)
        byr_cols = [col for col in data.columns if '_byrs' in col]
        data = feature_obj.byrs_count(data, all_train_data, col_list=byr_cols)
    except:
        print('{} 特征生产异常！'.format(sales_grid_id))
        return []

    # mapping处理
    for feature in map_list:
        code = 0
        mappings = {}
        for col in data[feature].unique():
            mappings[col] = code
            code += 1
        data[feature].replace(mappings, inplace=True)

    # 一级类目下的二级类目编码
    total_cat_data = []
    for cat1_name, cat_data in data.groupby('cat1_name'):
        code = 0
        mappings = {}
        for cat2_name in cat_data['cat2_name'].unique():
            mappings[cat2_name] = code
            code += 1
        cat_data['cat2_name'].replace(mappings, inplace=True)
        total_cat_data.append(cat_data)
    data = pd.concat(total_cat_data)

    # 精度控制
    for col in data.columns:
        _dtype = str(data[col].dtype)
        if 'float' in _dtype:
            data[col] = data[col].map(lambda x: round(x, 5))

    data.drop('dt', axis=1, inplace=True)

    # 数据量控制(避免图灵报错)
    if (data[data.is_train == 1].count()[0] < 2000) or \
            (data[data.is_train == 0].count()[0] < 5):
        return []

    # # 构造验证集数据("is_train"标记为3)
    # df_train = data[data.is_train == 1]
    # for cat in df_train.cat1_name.unique():
    #     cat_train = df_train[df_train.cat1_name == cat]
    #     ratio = cat_train.count()[0] * 0.05
    #     data.loc[df_train.sample(int(ratio)).index, 'is_train'] = 3

    for val in data.values:
        result_list.append(dict(zip(data.keys(), val)))

    return result_list


#############################################
# output data
#############################################
def write_data(df0, df1, configs):
    """Collect data locally and write to CSV.

    :param: rdd to print.
    :return: None
    """
    hive_path_t0 = str(configs['pricing_stragety_output_path'])+"/dt={}".format(str(configs['forecast_date']))
    hive_path_t1 = str(configs['pricing_stragety_output_path'])+"/dt={}".format(str(configs['forecast_end_date']))

    df0.repartition(10).write.mode('overwrite').orc(hive_path_t0)
    df1.repartition(1).write.mode('overwrite').orc(hive_path_t1)
    return None


#############################################
# entry point for PySpark application
#############################################
if __name__ == '__main__':
    main()