def model2pmml_string(model, model_name, X_columns, y_name):
    
    import os
    os.system('rm temp.pmml')
    
    # from sklearn2pmml import sklearn2pmml
    # sklearn2pmml(model, 'temp.pmml')
    
    from nyoka import skl_to_pmml, lgb_to_pmml, xgboost_to_pmml
    
    if model_name.startswith('lgb'):
        lgb_to_pmml(model, X_columns, y_name, "temp.pmml")
        
    elif model_name.startswith('xgb'):
        xgboost_to_pmml(model, X_columns, y_name, "temp.pmml")

    else:
        skl_to_pmml(model, X_columns, y_name, "temp.pmml")


    with open('temp.pmml') as f:
        lines = f.readlines()
    
    return ''.join([x.strip() for x in lines])

    
def save_pmml_to_hdfs(model, file_name, hdfs_path):
    """
    将训练结果
    """
    from sklearn2pmml import sklearn2pmml
    sklearn2pmml(model, file_name)
    
    import os
    os.system('hadoop fs -mkdir %s'%hdfs_path)
    
    put_to_hdfs(file_name, hdfs_path+file_name)
    
    os.system('rm %s'%file_name)

def put_to_hdfs(local_file, hdfs_path):
    """
    文件上传hdfs
    """
    from subprocess import PIPE, Popen
    put = Popen(["hadoop", "fs", "-put", local_file, hdfs_path], stdin=PIPE, bufsize=-1)
    put.communicate()
    
    
def extract_sku_info(df_sku, bu_id, wh_id, sku_id, target_hours, fc_dt = None, price_lag_num = 1, avg_sale_limit = 5):
    """
    输入单个仓+sku实时数据，生成对应特征
    target_hours:  以末尾几个小时的累计和作为目标，如21时预测，则可用数据为0-20, 需预测21时加22时销量和
    price_lag_num: 价格增长率的lag项
    fc_dt:         标识生成预测日期的数据
    """
    import pandas as pd
    import numpy as np
    
    assert df_sku.shape[0] > 0
    # sku上午成交量几乎为0, 不适合预测
    assert target_hours > 0 and target_hours < 12
    
    # 销量少的不参与训练，改mean为median
    if df_sku['arranged_cnt'].resample('D').sum()[-7:].median() < avg_sale_limit:
        return None

    # 小时粒度采样
    df_sku = df_sku.resample('H').sum().sort_index()
    
    if fc_dt:
        # 预测时只处理最新一段数据即可
        df_sku = df_sku[:fc_dt][-24*20:]
    
    # 区间销量特征
    df_sku_cnt = df_sku['arranged_cnt'].reset_index()
    df_sku_cnt['dt'] = df_sku_cnt['gap_btime'].apply(lambda x: x.strftime('%Y%m%d'))
    df_sku_cnt['hour'] = df_sku_cnt['gap_btime'].apply(lambda x: x.strftime('%H'))

    # 以日期为index, hour为列
    df_sku_cnt = pd.pivot_table(df_sku_cnt, index='dt',values='arranged_cnt', columns = 'hour').drop('23', axis = 1)
    df_sku_cnt.index = pd.to_datetime(df_sku_cnt.index)
    
    # 增加价格特征
    if price_lag_num > 0:
         # 每日售价特征
        df_sku_price = df_sku['arranged_amt'].resample('D').sum() / df_sku['arranged_cnt'].resample('D').sum()
    
        # 每天价格增长率
        price_rate = df_sku_price.diff()/df_sku_price.shift(1)
        
        price_rate_lag_list = []
        for i in range(price_lag_num):
            if i == 0:
                price_rate_lag_list.append(price_rate.rename('price_lag%d'%i))
            else:
                price_rate_lag_list.append(price_rate.shift(i).rename('price_lag%d'%i))

        df_sku_price_rate = pd.concat(price_rate_lag_list, axis = 1)

        df_features = pd.concat([df_sku_cnt, df_sku_price_rate], axis = 1)
    else:
        df_features = df_sku_cnt
    
    # bu_id, 仓id和sku id
    df_features['bu_id'] = bu_id
    df_features['wh_id'] = wh_id
    df_features['sku_id'] = sku_id
    
    # 增加日期相关特征
    # add_date_features(df_features)
    # df_features = add_date_feature(df_features)
    
    # 已销量和
    df_features['current_cnt_sum'] = df_features[['%02d'%x for x in range(23-target_hours)]].sum(axis=1)
    
    # 求和生成目标值
    drop_cols = [str(x) for x in range(22, 22-target_hours, -1)]
    
    df_features['y'] = df_features[drop_cols].sum(axis=1)

    # 目标值的lag项
    df_features['y_lag1'] = df_features['y'].shift(1)
    df_features['y_lag2'] = df_features['y'].shift(2)
    df_features['y_lag3'] = df_features['y'].shift(3)
    df_features['y_lag7'] = df_features['y'].shift(7)
    # df_features['y_lag14'] = df_features['y'].shift(14)
    
    # 同比增长？MA特征
    
    # 过去一段时间的平均值
    df_features['y_avg7'] = df_features['y'].shift(1).rolling(window=7).mean()
    df_features['y_std7'] = df_features['y'].shift(1).rolling(window=7).std()
    
    # df_features['y_avg14'] = df_features['y'].shift(1).rolling(window=14).mean()
    # df_features['y_std14'] = df_features['y'].shift(1).rolling(window=14).std()
    
    
    if fc_dt:
        # 生成预测数据
        df_features = df_features[[c for c in df_features if c not in drop_cols + ['y']]]

        # 无成交时该字段为空，填充0
        df_features['price_lag0'] = df_features['price_lag0'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return df_features.loc[fc_dt: fc_dt, :]

    else:
        # 生成训练数据, 将y作为最后一列
        df_features = df_features[[c for c in df_features if c != 'y'] + ['y']]

        # 目标值为0的不参与训练
        df_features = df_features[df_features['y'] > 0]

        return df_features.drop(drop_cols, axis = 1).replace([np.inf, -np.inf], np.nan).dropna()
    
    
def get_feature_array(data_rdd, columns, target_hours):
    """
    将sku真实销量转换为训练数据特征
    """
    import pandas as pd
    
    bu_id, wh_id, sku_id = data_rdd[0]
    
    # 构造预测维度原始df
    df = pd.DataFrame(data_rdd[1], columns = columns)
    df['gap_btime'] = pd.to_datetime(df['gap_btime'])
    df = df.set_index('gap_btime')
        
    try:
        # 生成特征df
        df_features = extract_sku_info(df, bu_id, wh_id, sku_id, target_hours=target_hours)
    except:
        # 数据不全不参与训练
        return [None]

    return [None] if df_features is None or df_features.shape[0]==0 else df_features.values.tolist()


def train_model(spark, df_raw, columns, target_hours=4):
    """
    根据预测时段训练多个模型
    """
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DoubleType
    
    rdd_colleted = df_raw.rdd.map(lambda row: ((row['bu_id'], row['wh_id'], row['sku_id']), list(row)))\
                                .groupByKey().flatMap(lambda row: get_feature_array(row, columns, target_hours=target_hours))\
                                .filter(lambda x: x is not None).collect()

    assert len(rdd_colleted) > 0
    feature_num = len(rdd_colleted[0])
    

    schema = StructType([StructField("F%d"%i, FloatType(), True) for i in range(feature_num-1)] + 
                        [StructField("Y", FloatType(), True)])
    
    df_features = spark.createDataFrame(rdd_colleted, schema).toPandas()

    X_train = df_features.iloc[:, :-1]
    y_train = df_features.iloc[:, -1]
    
    print('训练集X_train.shape=', X_train.shape)
    
    
    from sklearn.pipeline import Pipeline
    
    # 训练结果
    pipeline_dict = {}
    
    # # 训练lgb
    # try:
    #     import lightgbm as lgb
    #     pipeline_dict['lgb'] = Pipeline([
    #         ("regressor", lgb.LGBMRegressor(objective='regression', num_leaves=30,
    #                                   learning_rate=0.1, n_estimators=10, max_depth=5, 
    #                                   metric='mse', subsample = 0.9, colsample_bytree = 0.9))
    #     ]).fit(X_train, y_train)
    # except:
    #     pass
    
    # 训练rf
    # try:
    #     from sklearn.ensemble import RandomForestRegressor
    #     pipeline_dict['rf_realtime_3p'] = Pipeline([
    #         ("regressor", RandomForestRegressor())
    #     ]).fit(X_train, y_train)
    # except:
    #     pass

    # # 训练etr
    # try:
    #     from sklearn.ensemble import ExtraTreesRegressor
    #     pipeline_dict['etr_realtime_3p'] = Pipeline([
    #         ("regressor", ExtraTreesRegressor())
    #     ]).fit(X_train, y_train)
    # except:
    #     pass
    
    # xgb
    try:
        from xgboost.sklearn import XGBRegressor
        pipeline_dict['xgb_realtime_3p'] = Pipeline([
            ("regressor", XGBRegressor())
        ]).fit(X_train, y_train)
    except:
        pass

    # 训练gbdt
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        pipeline_dict['gbdt_realtime_3p'] = Pipeline([
            ("regressor", GradientBoostingRegressor())
        ]).fit(X_train, y_train)
    except:
        pass

    print('pipeline_dict.keys() = ', pipeline_dict.keys())
    
    return pipeline_dict, X_train.columns, y_train.name
    
