import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

"""
量价模型-特征数据预处理-工具类
"""

class DataUtil:

    def __init__(self, forecast_date):
        self.forecast_date = forecast_date

    def sku_to_train(self, df_raw):
        """
        返回近30天有售卖的sku
        """

        b_1d_fc_dt = pd.to_datetime(self.forecast_date) - timedelta(1)
        b_30d_fc_dt = pd.to_datetime(self.forecast_date) - timedelta(30)

        sku_30d_arr = df_raw[(df_raw['dt'] <= b_1d_fc_dt) & (df_raw['dt'] >= b_30d_fc_dt)]
        sku_30d_arr_sum = sku_30d_arr.groupby('sku_id', as_index=False)[['arranged_cnt']].sum()
        sku_train_list = sku_30d_arr_sum[sku_30d_arr_sum['arranged_cnt'] != 0]['sku_id'].tolist()

        return sku_train_list

    def preprocess(self, df_raw, is_train=True):
        """
        预处理函数
        """

        df_raw = df_raw.sort_values('dt')

        # 处理brand特征
        df_raw.loc[df_raw['brand_name'] != '无', 'brand_name'] = 1
        df_raw.loc[df_raw['brand_name'] == '无', 'brand_name'] = 0


        # 价格异常数据修正

        ## 异常1: 售价<促销价
        sprice_odd_list = df_raw[df_raw['price'] < df_raw['seq_price']].index.tolist()
        pprice_odd_list = df_raw[df_raw['price'] < df_raw['pro_price']].index.tolist()
        df_raw.loc[sprice_odd_list, 'seq_price'] = df_raw.loc[sprice_odd_list, 'price']
        df_raw.loc[sprice_odd_list, 'seq_num'] = 0
        df_raw.loc[pprice_odd_list, 'pro_price'] = df_raw.loc[pprice_odd_list, 'price']
        df_raw.loc[pprice_odd_list, 'pro_num'] = 0

        ## 异常2: 售价=促销价&促销场次数>0
        pnum_odd_list = df_raw[(df_raw['price'] == df_raw['pro_price']) &
                               (df_raw['pro_num'] > 0)].index.tolist()
        snum_odd_list = df_raw[(df_raw['price'] == df_raw['seq_price']) &
                               (df_raw['seq_num'] > 0)].index.tolist()
        df_raw.loc[pnum_odd_list, 'pro_num'] = 0
        df_raw.loc[snum_odd_list, 'seq_num'] = 0


        # 处理天气缺失值&异常值

        ## step1: 用空值替换气温异常值
        df_raw.loc[(df_raw['avg_t'] > 100) | (df_raw['avg_t'] < -100), 'avg_t'] = np.nan
        df_raw.loc[(df_raw['avg_t_b'] > 100) | (df_raw['avg_t_b'] < -100), 'avg_t_b'] = np.nan

        ## step2: 用月份平均气温填补空值
        df_raw['month'] =  pd.to_datetime(df_raw['dt']).dt.month
        month_temp = df_raw.groupby('month', as_index=False)['avg_t'].agg({'month_temp': np.mean})
        df_raw = df_raw.merge(month_temp, how='left', on=['month'])
        t_nan = df_raw[df_raw['avg_t'].isnull() == True].index
        tb_nan = df_raw[df_raw['avg_t_b'].isnull() == True].index
        df_raw.loc[t_nan, 'avg_t'] = df_raw.loc[t_nan, 'month_temp']
        df_raw.loc[tb_nan, 'avg_t_b'] = df_raw.loc[tb_nan, 'month_temp']

        ## step3: 整体填补&去掉冗余列
        df_raw[['avg_t', 'avg_t_b']] = df_raw[['avg_t', 'avg_t_b']].fillna(method='bfill')\
            .fillna(method='ffill')\
            .fillna(0.)
        df_raw = df_raw.iloc[:, :-2]


        # 处理商户缺失值
        byr_cols = [col for col in df_raw.columns if '_byrs' in col]
        for col in byr_cols:
            df_raw[col] = df_raw[col].fillna(method='bfill')
            df_raw[col] = df_raw[col].fillna(method='ffill')
            df_raw[col].fillna(0., inplace=True)


        # 数据截取&空值处理
        if is_train:
            df_raw = df_raw[df_raw.dt < pd.to_datetime(self.forecast_date)]
            df_raw = df_raw.dropna()
        else:
            df_raw = df_raw[df_raw.dt >= pd.to_datetime(self.forecast_date)]

        return df_raw