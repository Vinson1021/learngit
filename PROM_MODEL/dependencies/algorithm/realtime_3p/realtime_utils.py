def model2pmml_string(model, model_name, X_columns, y_name):
    
    import os
    os.system('rm temp.pmml')

    # import sys
    # os.environ['PYTHONPATH'] = './PMML_PKG/pmml_pkgs:' + os.environ['PYTHONPATH']
    # sys.path.insert(0, './PMML_PKG/pmml_pkgs')

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
    
    
def add_date_feature(df):

    import math
    import numpy as np
    df = df.reset_index()
    # df['minute'] = df['index'].apply(lambda x: x.hour*60 + x.minute)
    # df["hour"] = df['index'].dt.hour.astype(np.int8)
    # df["weekend"] = df[index_name].dt.weekday.astype(np.int8) # 与dayofweek相同
    df['month'] = df['index'].dt.month.astype(np.int8)
    df['dayofweek'] = df['index'].dt.dayofweek.astype(np.int8)
    df['dayofmonth'] = df['index'].dt.day.astype(np.int8)
    df['dayofyear'] = df['index'].dt.dayofyear.astype(np.int16)
    df['weekofmonth'] = (df['index'].dt.day/7).apply(lambda x: math.ceil(x)).astype(np.int8)
    df['weekofyear'] = (df['dayofyear']/7).apply(lambda x: math.ceil(x)).astype(np.int8)
    # df['is_holiday']= df['index'].map(is_holiday).astype(np.int8)
    
    return df.set_index('index').sort_index()
    
def extract_sku_info(df_sku, bu_id, wh_id, sku_id, target_hours, fc_dt = None, price_lag_num = 1, 
                     avg_sale_limit = 10, pred_hist_days = 20):
    """
    输入单个仓+sku实时数据，生成对应特征
    target_hours:   以末尾几个小时的累计和作为目标，如21时预测，则可用数据为0-20, 需预测21时加22时销量和
    price_lag_num:  价格增长率的lag项
    fc_dt:          标识生成预测日期的数据
    pred_hist_days: 预测时有效数据历史天数
    """
    import pandas as pd
    import numpy as np
    
    assert df_sku.shape[0] > 0
    # sku上午成交量几乎为0, 不适合预测
    assert target_hours > 0 and target_hours < 12
    
    # 销量少的不参与训练，改mean为median
    if df_sku['arranged_cnt'].resample('D').sum().sort_index()[-8:-1].median() < avg_sale_limit:
        return None

    # 小时粒度采样
    df_sku = df_sku.resample('H').sum().sort_index()
    
    if fc_dt:
        # 预测时只处理最新一段数据即可
        df_sku = df_sku[:fc_dt][-24*pred_hist_days:]
    
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
        # price_rate = df_sku_price.diff()/df_sku_price.shift(1)
        price_rate = df_sku_price.diff()/df_sku_price.shift(1).rolling(window=7).mean()
        
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
    df_features = add_date_feature(df_features)
    
    # 已销量和
    df_features['current_cnt_sum'] = df_features[['%02d'%x for x in range(23-target_hours)]].sum(axis=1)
    
    # 求和生成目标值
    drop_cols = [str(x) for x in range(22, 22-target_hours, -1)]
    
    df_features['y'] = df_features[drop_cols].sum(axis=1)

    # 目标值的lag项
    df_features['y_lag1'] = df_features['y'].shift(1)
    df_features['y_lag2'] = df_features['y'].shift(2)
    df_features['y_lag3'] = df_features['y'].shift(3)
    df_features['y_lag4'] = df_features['y'].shift(4)
    df_features['y_lag5'] = df_features['y'].shift(5)
    df_features['y_lag6'] = df_features['y'].shift(6)
    df_features['y_lag7'] = df_features['y'].shift(7)
    # df_features['y_lag14'] = df_features['y'].shift(14)
    
    # 同比增长？MA特征
    
    # 过去一段时间的平均值
    df_features['y_avg7'] = df_features['y'].shift(1).rolling(window=7).mean()
    df_features['y_std7'] = df_features['y'].shift(1).rolling(window=7).std()
    df_features['y_median7'] = df_features['y'].shift(1).rolling(window=7).median()
    df_features['y_min7'] = df_features['y'].shift(1).rolling(window=7).min()
    df_features['y_max7'] = df_features['y'].shift(1).rolling(window=7).max()
    
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

        # 训练集中删除
        df_features = df_features[df_features['current_cnt_sum'] > 0]

        return df_features.drop(drop_cols, axis = 1).replace([np.inf, -np.inf], np.nan).dropna()
    
    
def get_feature_array(row, columns):
    """
    将sku真实销量转换为训练数据特征
    """
    import pandas as pd
    
    bu_id, wh_id, sku_id, hour, model_name = row[0]
    target_hours = 23 - hour
    
    # 构造预测维度原始df
    df = pd.DataFrame(row[1], columns = columns)
    df['gap_btime'] = pd.to_datetime(df['gap_btime'])
    df = df.set_index('gap_btime')
        
    try:
        # 生成特征df
        df_features = extract_sku_info(df, bu_id, wh_id, sku_id, target_hours=target_hours)
    except:
        # 数据不全不参与训练
        return [None]

    return [None] if df_features is None or df_features.shape[0]==0 else [(hour, model_name), df_features.values.tolist()]

def combine_features(f1, f2):
    """
    将两个特征矩阵合并
    """
    import numpy as np
    f1 = np.array(f1)
    f2 = np.array(f2)
    return np.vstack((f1, f2)).tolist()
    
def pmml_encode(pmml_str):
    """
    返回压缩并base64编码的pmml结果
    """
    import zlib
    import base64
    return base64.b64encode(zlib.compress(pmml_str.encode())).decode()
    
    
def pmml_decode(base64_str):
    """
    解码base64并解压，返回原始pmml
    """
    import zlib
    import base64
    return zlib.decompress(base64.b64decode(base64_str)).decode()


def train_and_output_pmml(X_train, y_train, model_name):
    """
    依据模型训练并返回pmml
    """
    from sklearn.pipeline import Pipeline
    if model_name.startswith('rf'):
        # 训练rf
        try:
            from sklearn.ensemble import RandomForestRegressor
            pipeline = Pipeline([
                ("regressor", RandomForestRegressor())
            ]).fit(X_train, y_train)
        except:
            return None

    elif model_name.startswith('etr'):
        # 训练etr
        try:
            from sklearn.ensemble import ExtraTreesRegressor
            pipeline = Pipeline([
                ("regressor", ExtraTreesRegressor())
            ]).fit(X_train, y_train)
        except:
            return None

    elif model_name.startswith('xgb'):
        # xgb
        try:
            from xgboost.sklearn import XGBRegressor
            pipeline = Pipeline([
                ("regressor", XGBRegressor(n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree = 0.8,
                                            max_depth = 5, min_child_weight=1))
            ]).fit(X_train, y_train)

        except:
            return None


    elif model_name.startswith('gbdt'):
        # 训练gbdt
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            pipeline = Pipeline([
                ("regressor", GradientBoostingRegressor())
            ]).fit(X_train, y_train)

        except:
            return None

    elif model_name.startswith('dnn'):
        # DNN
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.neural_network import MLPRegressor
            pipeline = Pipeline([
                        ("standardize", StandardScaler()),
                        ("regressor", MLPRegressor(solver='adam', hidden_layer_sizes=(5, 10, 5), early_stopping=True, random_state=10))
                    ]).fit(X_train, y_train)
        except:
            return None

    else:
        return None

    print('模型训练结束!')
    print(model_name, pipeline)
    # return model2pmml_string(pipeline, model_name, X_train.columns, y_train.name)

    try:
        pmml = pmml_encode(model2pmml_string(pipeline, model_name, X_train.columns, y_train.name))
    except:
        return None
    else:
        return pmml

def train_model(row):
    """
    根据预测时段训练多个模型
    """
    import pandas as pd
    import numpy as np
    import datetime
    
    dt = datetime.datetime.now().strftime("%Y%m%d")
    hour, model_name = row[0]
    train_data = np.array(row[1])
    
    row_num, col_num = train_data.shape[0], train_data.shape[1]
    
    assert row_num > 0 and col_num > 0
    
    df_features = pd.DataFrame(train_data, columns=['F%d'%x for x in range(col_num-1)] + ['Y'])
    
    
    X_train = df_features.iloc[:, :-1]
    y_train = df_features.iloc[:, -1]
    
    # print('训练集X_train.shape=', X_train.shape)
    
    pmml = train_and_output_pmml(X_train, y_train, model_name)
    
    return [None] if pmml is None else [dt, model_name, 'PMML', hour, pmml]
    

