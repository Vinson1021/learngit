#!/bin/usr/env python3.6
# -*- coding: utf-8 -*-
# ---------------------------------
# ProjectName: data-forecast-spark
# Description: 促销模型预处理函数
# Author: guoyunshen
# CreateTime: 2020-10-21 10:52
# ---------------------------------

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')


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
        df = df_raw.sort_values('dt')

        # 处理brand特征
        df.loc[df['brand_name'] != '无', 'brand_name'] = 1
        df.loc[df['brand_name'] == '无', 'brand_name'] = 0

        # 价格异常数据修正

        ## 异常1: 售价<促销价
        sprice_odd_list = df[df['price'] < df['seq_price']].index.tolist()
        pprice_odd_list = df[df['price'] < df['pro_price']].index.tolist()
        df.loc[sprice_odd_list, 'seq_price'] = df.loc[sprice_odd_list, 'price']
        df.loc[sprice_odd_list, 'seq_num'] = 0
        df.loc[pprice_odd_list, 'pro_price'] = df.loc[pprice_odd_list, 'price']
        df.loc[pprice_odd_list, 'pro_num'] = 0

        ## 异常2: 售价=促销价&促销场次数>0
        pnum_odd_list = df[(df['price'] == df['pro_price']) &
                           (df['pro_num'] > 0)].index.tolist()
        snum_odd_list = df[(df['price'] == df['seq_price']) &
                           (df['seq_num'] > 0)].index.tolist()
        df.loc[pnum_odd_list, 'pro_num'] = 0
        df.loc[snum_odd_list, 'seq_num'] = 0

        ## 测试集用最低价填补'w_price'字段
        if not is_train:
            df['w_price'] = df[['price', 'seq_price', 'pro_price']].min(axis=1)

        # 处理天气缺失值&异常值

        ## step1: 用空值替换气温异常值
        df.loc[(df['avg_t'] > 100) | (df['avg_t'] < -100), 'avg_t'] = np.nan
        df.loc[(df['avg_t_b'] > 100) | (df['avg_t_b'] < -100), 'avg_t_b'] = np.nan

        ## step2: 用月份平均气温填补空值
        df['month'] = pd.to_datetime(df['dt']).dt.month
        month_temp = df.groupby('month', as_index=False)['avg_t'].agg({'month_temp': np.mean})
        df = df.merge(month_temp, how='left', on=['month'])
        t_nan = df[df['avg_t'].isnull() == True].index
        tb_nan = df[df['avg_t_b'].isnull() == True].index
        df.loc[t_nan, 'avg_t'] = df.loc[t_nan, 'month_temp']
        df.loc[tb_nan, 'avg_t_b'] = df.loc[tb_nan, 'month_temp']

        ## step3: 整体填补&去掉冗余列
        df[['avg_t', 'avg_t_b']] = df[['avg_t', 'avg_t_b']].fillna(method='bfill') \
            .fillna(method='ffill') \
            .fillna(0.)
        df = df.iloc[:, :-2]

        # 处理商户缺失值
        byr_cols = [col for col in df.columns if '_byrs' in col]
        for col in byr_cols:
            df[col] = df[col].fillna(method='bfill')
            df[col] = df[col].fillna(method='ffill')
            df[col].fillna(0., inplace=True)

        # 数据截取&空值处理
        if is_train:
            df = df[df.dt < pd.to_datetime(self.forecast_date)]
            df = df.dropna()
        else:
            df = df[df.dt >= pd.to_datetime(self.forecast_date)]

        return df