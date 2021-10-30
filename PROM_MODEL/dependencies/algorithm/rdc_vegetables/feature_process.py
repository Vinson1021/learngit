import pandas as pd
import numpy as np
import datetime
import json
import pickle
from functools import partial
from dependencies.algorithm.rdc_vegetables.predict_algo import moving_avg, ses
# from predict_algo import *

def within_holiday_windows(dt):
    """
    在长假前一天及长假期间标识
    一般销量提前一天已经进入法定假期模式
    """
    return int(is_holiday(dt) or is_holiday(dt + pd.Timedelta(days = 1)))
    
def is_holiday(dt):
    """
    法定长假
    """
    
    # 2020年法定假日
    holidays = [

            # 2019年节假日
            '2019-01-01',  # 元旦
            '2019-02-04', '2019-02-05', '2019-02-06', '2019-02-07', '2019-02-08', '2019-02-09', '2019-02-10',  # 春节
            '2019-04-05', '2019-04-06', '2019-04-07', # 清明
            '2019-05-01', '2019-05-02', '2019-05-03', '2019-05-04', # 五一
            '2019-06-07', '2019-06-08', '2019-06-09', # 端午
            '2019-09-13', '2019-09-14', '2019-09-15', # 中秋节
            '2019-10-01', '2019-10-02', '2019-10-03', '2019-10-04', '2019-10-05', '2019-10-06', '2019-10-07', # 十一

            # 2020年节假日
            '2020-01-01',  # 元旦
            '2020-01-24', '2020-01-25', '2020-01-26','2020-01-27', '2020-01-28', '2020-01-29', '2020-01-30', 
            '2020-01-31', '2020-02-01', '2020-02-02', # 春节
            '2020-04-04', '2020-04-05', '2020-04-06', # 清明
            '2020-05-01', '2020-05-02', '2020-05-03', '2020-05-04', '2020-05-05', # 五一
            '2020-06-25', '2020-06-26', '2020-06-27', # 端午
            '2020-10-01', '2020-10-02', '2020-10-03', '2020-10-04', '2020-10-05', '2020-10-06', 
            '2020-10-07', '2020-10-08',  # 国庆

            # 2021年节假日
            '2021-01-01', '2021-01-02', '2021-01-03', # 元旦
            '2021-02-11', '2021-02-12', '2021-02-13', '2021-02-14', '2021-02-15', '2021-02-16', '2021-02-17', # 春节
            '2021-04-03', '2021-04-04', '2021-04-05', # 清明
            '2021-05-01', '2021-05-02', '2021-05-03', '2021-05-04', '2021-05-05', # 五一
            '2021-06-12', '2021-06-13', '2021-06-14', # 端午
            '2021-09-19', '2021-09-20', '2021-09-21', # 中秋节
            '2021-10-01', '2021-10-02', '2021-10-03', '2021-10-04', '2021-10-05', '2021-10-06', '2021-10-07' # 十一

            ]
    
    # assert isinstance(dt, datetime.datetime)
    # assert dt.strftime('%Y') in ('2019', '2020', '2021')

    return dt.strftime('%Y-%m-%d') in holidays

def weekday(dt):
    return dt.weekday()

def monthday(dt):
    return dt.day

def is_week_day(weekday, dt):
    """
    返回星期标识, 周一为0
    """
    return 1 if weekday == dt.weekday() else 0  
    
def nextday_is_workday(dt):
    """
    前一天为工作日
    """
    return is_workday(dt + pd.Timedelta(days = 1))
    
def is_workday(dt):
    """
    是否为工作日，法定假日返回1，否则为0
    """

    # 2020年法定假日调整后，某些周末改为工作日
    workdays = [
                    '2019-02-02', '2019-02-03', '2019-04-28', '2019-05-05', '2019-09-29', '2019-10-12',
                    '2020-01-19', '2020-04-26', '2020-05-09', '2020-06-28', '2020-09-27', '2020-10-10',
                    '2021-02-07', '2021-02-20', '2021-04-25', '2021-05-08', '2021-09-18', '2021-09-26', '2021-10-09'
                ]
    
    # assert isinstance(dt, datetime.datetime)
    # assert dt.strftime('%Y') in ('2019', '2020', '2021')
    
    if dt.weekday() in (5, 6) and dt.strftime('%Y-%m-%d') not in workdays:
        # 非调休的周末
        return 0

    elif is_holiday(dt):
        # 法定节假日
        return 0

    else:
        return 1
    
def series_to_supervised(data, n_in = 14, n_out=1, dropnan = True, dropzero = True):
    """
    将Series或numpy转换为特征矩阵
    n_in:  lag数量
    n_out: 预测步长
    dropnan: 是否删除nan值
    """
    n_vars = 1 if isinstance(data, pd.Series) or isinstance(data, list) else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    # 输入预测(t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    # 预测步长 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
        
    if dropzero:
        agg = agg[(agg != 0).all(axis=1)]
        # agg = agg[(agg != 0).any(axis=1)]
    return agg

def filter_big_promo(y):
    """
    过滤大促异常值
    """
    assert isinstance(y, pd.Series)
    
    # 过去3天销量求和作为门限
    last3_sum = y.shift(1).rolling(window=3).sum()

    # 预测前一天超过门限日期则赋值为平均值
    big_promo_date = y > last3_sum

    # 大促异常值赋为均值
    y[big_promo_date] = last3_sum[big_promo_date]/3

def adding_features(df):
    """
    对已经处理为df格式的特征进行加工
    """
    assert df.shape[0] > 0
    
    # 对前一天超量情况进行特殊处理
    # 过去3天销量求和作为门限
    # last3_sum = df.loc[:, 'var1(t-4)':'var1(t-2)'].sum(axis = 1)

    # 预测前一天超过门限日期则赋值为平均值
    # big_promo_date = df.loc[:, 'var1(t-1)'] > last3_sum

    # df.loc[big_promo_date, 'var1(t-1)'] = last3_sum[big_promo_date]/3
    
    # 增加日期相关特征
    df['workday_t0'] = df.index.map(is_workday)

    df['workday_t1'] = df.index.map(nextday_is_workday)
    
    # 星期用一个值
    # df['weekday'] = df.index.map(weekday)
    
    # 星期特征, 使用onehot编码, 周一对应w0为1
    for weekday in range(7):
        fun = partial(is_week_day, weekday) 
        df['w%d'%weekday] = df.index.map(fun)
    
    # 法定假日标识
    df['holiday_t0'] = df.index.map(within_holiday_windows)
    
    # onehot与均值交叉
    lag_14_median = df.apply(lambda r: np.median(r[:14]), axis = 1)
    for col_name in ('workday_t0', 'workday_t1', 'w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'holiday_t0'):
        df[col_name] *= lag_14_median
    
    return df.astype(int)

def get_feature_array(data_rdd, columns, lag = 14):
    """
    将sku真实销量转换为训练数据特征
    """
    bu_id, wh_id, sku_id = data_rdd[0]
    
    df = pd.DataFrame(data_rdd[1], columns = columns)

    df['dt'] = pd.to_datetime(df['dt'])
    df = df.set_index('dt')

    y = df['arranged_cnt']

    y = y.sort_index()
    
    # 此处填充s中空的日期
    y = y.resample('D').asfreq().bfill().ffill()

    df_feature = series_to_supervised(y, n_in = lag)
    
    if df_feature.shape[0] == 0:
        return [None]
    
    # 过滤大促异常值
    filter_big_promo(y)
    
    # 对特征进行加工
    df_feature = adding_features(df_feature)
    
    # 将bu_id, wh_id, sku_id加入特征中
    df_feature['bu_id'] = str(bu_id)
    df_feature['wh_id'] = str(wh_id)
    df_feature['sku_id'] = str(sku_id)
    
    # 将预测值列移动到最后一列位置
    df_feature = df_feature[[c for c in df_feature if c != 'var1(t)'] + ['var1(t)']]
    
    return df_feature.values.tolist()

def extract_train_data(spark):

    sql = """
    select * from 
    (
      select t1.dt, t1.bu_id, t1.wh_id, t1.sku_id, t2.cat1_id, t1.arranged_cnt, t1.on_shelf, t1.able_sell from
      mart_caterb2b_forecast.app_caterb2b_forecast_input_sales_dt_wh_sku t1
      left join mart_caterb2b.dim_caterb2b_sku t2
      on t1.sku_id = t2.sku_id
    )
    where dt >='20200101'
    and able_sell = 1
    and cat2_id = 10021362
    and wh_id in (189, 101, 100, 263, 112, 82, 251, 65, 192, 332) 
    """
    return spark.sql(sql)
    
def get_train_model(data_rdd, columns):
    """
    集中式模型训练
    """
    
    rdd_colleted = data_rdd.map(lambda row: ((row['bu_id'], row['wh_id'], row['sku_id']), list(row)))\
                                .groupByKey().flatMap(lambda row: get_feature_array(row, columns))\
                                .filter(lambda x: x is not None).collect()
        
    data = np.array(rdd_colleted)
    
    # 可以选择是否将bu_id, wh_id, sku_id作为特征
    X_train = data[:, :-4].astype(np.int32)   # X_train = data[:, :-1]
    y_train = data[:, -1:].astype(np.int32).reshape(-1)
    
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    from sklearn.ensemble import GradientBoostingRegressor
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train, y_train)
    
    from xgboost.sklearn import XGBRegressor
    xgb = XGBRegressor()
    xgb.fit(X_train,y_train)

    return {'lr': lr, 'gbr': gbr, 'xgb': xgb}

def rdd_train(data_rdd, columns):
    """
    在RDD上训练模型
    如果训练模型数量过多，会在collect时超过1G容量，可以配置spark调高spark.driver.maxResultSize
    """

    group_id = data_rdd[0]
    
    df = pd.DataFrame(data_rdd[1], columns = columns)
    
    # 提供相关特征
    df = df[[x for x in df.columns if x[0] == 'F' or x == 'Y']]
    
    data = df.values
    X_train = data[:, :-1]
    y_train = data[:, -1:].reshape(-1)
    
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    from sklearn.ensemble import ExtraTreesRegressor
    etr = ExtraTreesRegressor(n_estimators = 200)
    etr.fit(X_train,y_train)
    
#     from sklearn.ensemble import GradientBoostingRegressor
#     gbr = GradientBoostingRegressor(n_estimators=250, max_depth=3,
#                                 learning_rate=.01, min_samples_leaf=9,
#                                 min_samples_split=9)
#     gbr.fit(X_train, y_train)
    
    
    return (group_id, df.shape[0], df.shape[1], {
                                                    'lr': pickle.dumps(lr), 
                                                    'etr': pickle.dumps(etr), 
                                                    # 'gbr': pickle.dumps(gbr)
                                                })

def train_model(df_raw, spark):
    """
    输入训练数据RDD，返回训练后的模型字典
    """
    columns = df_raw.columns
    
    # 特征生成
    rdd_features = df_raw.rdd.map(lambda row: ((row['bu_id'], row['wh_id'], row['sku_id']), list(row)))\
                                    .groupByKey().flatMap(lambda row: get_feature_array(row, columns))\
                                    .filter(lambda x: x is not None)
    
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

    # 特征结构
    schema = StructType([
        StructField("F1", IntegerType(), True),
        StructField("F2", IntegerType(), True),
        StructField("F3", IntegerType(), True),
        StructField("F4", IntegerType(), True),
        StructField("F5", IntegerType(), True),
        StructField("F6", IntegerType(), True),
        StructField("F7", IntegerType(), True),
        StructField("F8", IntegerType(), True),
        StructField("F9", IntegerType(), True),
        StructField("F10", IntegerType(), True),
        StructField("F11", IntegerType(), True),
        StructField("F12", IntegerType(), True),
        StructField("F13", IntegerType(), True),
        StructField("F14", IntegerType(), True),
        StructField("F15", IntegerType(), True),
        StructField("F16", IntegerType(), True),
        StructField("F17", IntegerType(), True),
        StructField("F18", IntegerType(), True),
        StructField("F19", IntegerType(), True),
        StructField("F20", IntegerType(), True),
        StructField("F21", IntegerType(), True),
        StructField("F22", IntegerType(), True),
        StructField("F23", IntegerType(), True),
        StructField("bu_id", StringType(), True),
        StructField("wh_id", StringType(), True),
        StructField("sku_id", StringType(), True),
        StructField("Y", IntegerType(), True)
    ])

    # 特征df
    df_features = spark.createDataFrame(rdd_features, schema)

    # 在RDD上训练模型
    columns =  df_features.columns
    
    model_trained = df_features.rdd.map(lambda row: (row['wh_id'], list(row)))\
                                    .groupByKey().map(lambda row: rdd_train(row, columns))\
                                    .filter(lambda x: x is not None).collect()
    # 返回训练后的模型字典
    return {int(group_id): model_dict for group_id, _, _, model_dict in model_trained}

def get_sku_latest_feature(y, bu_id, wh_id, sku_id, lag = 14):
    """
    返回sku的最新特征数据，用于生成预测结果
    """
    assert isinstance(y, pd.Series)

    y = y[-lag:]
    
    # 过滤大促异常值
    filter_big_promo(y)

    df_feature = pd.DataFrame(y.values.reshape(1, -1), index = [y.index[-1]+ pd.Timedelta(days=1)], 
                              columns = ['var1(t-%d)'%i for i in range(14, 0, -1)])

    # 对特征进行加工
    df_feature = adding_features(df_feature)
    
    # 将bu_id, wh_id, sku_id加入特征中
    # df_feature['bu_id'] = str(bu_id)
    # df_feature['wh_id'] = str(wh_id)
    # df_feature['sku_id'] = str(sku_id)

    return df_feature.iloc[-1, :].values


def sku_predict_t0(model, y, fc_dt, param_dict,ndays = 20, lack_days_limit = 7):
    """
    预测fc_dt天t0结果
    允许从昨天以前缺历史数据的天数，lack_days_limit
    外推天数过长容易过拟合趋势性变化
    """
    assert isinstance(y, pd.Series) and len(y) > 0
    
    y = y.sort_index()
    y = y.resample('D').asfreq().bfill().ffill()
    
    # 仅保留预测日期前的数据
    last_valid_day = pd.to_datetime(fc_dt)-pd.Timedelta(days=1)
    y = y[:last_valid_day]
    
    recursive_step = (last_valid_day - y.index[-1]).days + 1
    
    assert recursive_step <= lack_days_limit 
    
    for i in range(recursive_step):
        
        # 生成最新一条特征数据
        feature = get_sku_latest_feature(y, param_dict['bu_id'], param_dict['wh_id'], param_dict['sku_id'])
        
        # 确保生成历史销量数据
        assert feature.shape[0] != 0

        # 调用模型预测下一天结果
        y_pred = model.predict(feature.reshape(1, -1))[0]

        # 将预测结果循环加入
        y = y.append(pd.Series([y_pred], index=[y.index[-1] + pd.Timedelta(days = 1)]))
        
    y_index = pd.date_range(pd.to_datetime(fc_dt), periods = ndays, freq='D')
    
    return round(pd.Series([y_pred] * ndays, index=y_index), 2)

def post_process(y):
    """
    预测结果后处理逻辑
    """
    # 每天预测值非负
    y[y < 0] = 0
    y[y > 3*y.mean()] = 3*y.mean()


def predict(data_rdd, columns, fc_dt, model, model_name, model_id, ndays = 1):
    """
    在RDD上调用预测算法
    fc_dt:      预测日期
    model_name: 算法名称，支持以下算法: ses, moving_avg, lr, gbdt, xgb
    """

    bu_id, wh_id, sku_id = data_rdd[0]
    param_dict = {'bu_id': bu_id, 'wh_id': wh_id, 'sku_id': sku_id}
    
    df = pd.DataFrame(data_rdd[1], columns = columns)

    df['dt'] = pd.to_datetime(df['dt'])
    df = df.set_index('dt')
    df = df.sort_index()

    y = df['arranged_cnt']

    if len(y) == 0:
        return None

    if model_name == 'ses':
        # 简单指数回归
        y_pred = ses(y, fc_dt, ndays)
        
    elif model_name == 'moving_avg':
        # 加权平均
        y_pred = moving_avg(y, fc_dt, ndays)

    else:
        # GBR, XGB, LR等模型
        try:
            y_pred = sku_predict_t0(model, y, fc_dt, param_dict, ndays)
            # y_pred = sku_predict_t0(pickle.loads(model[wh_id][model_name]), y, fc_dt, ndays)

        except Exception as e:
            print(e)
            y_pred = ses(y, fc_dt, ndays)
        
    if y_pred is None:
        return None

    else:
        # 预测结果后处理
        post_process(y_pred)
        
        y_pred.index = [x.strftime("%Y%m%d") for x in y_pred.index]
        return (fc_dt, bu_id, wh_id, sku_id, 0, model_id, float(y_pred.sum()), json.dumps(y_pred.to_dict()))
    
    