#!/bin/usr/env python3.6
# -*- coding: utf-8 -*-
# ---------------------------------
# ProjectName: data-forecast-spark
# Description: 3P销量预测数据预处理
# Author: songzhen07
# CreateTime: 2021-04-14 14:37
# ---------------------------------

import pandas as pd
import numpy as np
import random
random.seed(42)
np.random.seed(42)

import warnings
warnings.filterwarnings('ignore')



class SalePopDataPreprocess:

    def __init__(self, forecast_date):
        self.forecast_date = forecast_date


    def prepro_func(self, df_raw, is_train=True):
        """
        预处理函数
        """
        df = df_raw.sort_values(['dt', 'hour'])
        nonan_data = []


        # 处理brand特征
        df['brand_name'].fillna('无', inplace=True)


        # 原价缺失值
        for key, data in df.groupby(['bu_id', 'sku_id']):
            nonan_data.append(data[['origin_price']] \
                              .fillna(method='ffill') \
                              .fillna(method='bfill'))
        df_nonan = pd.concat(nonan_data).reindex(index=df.index)
        df[['origin_price']] = df_nonan


        ## 用原价填补(剩余)缺失的促销价格
        mprice_nan = df[df['price'].isnull() == True].index
        df.loc[mprice_nan, 'price'] = df.loc[mprice_nan, 'origin_price']

        # 原价/促销价数据修正
        ## 售价低于促销价
        sprice_odd_list = df[df['origin_price'] < df['price']].index.tolist()
        df.loc[sprice_odd_list, 'price'] = df.loc[sprice_odd_list, 'origin_price']


        # 日期处理
        #构造日期特征
        df['year'] = pd.to_datetime(df['dt']).dt.year
        df['month'] = pd.to_datetime(df['dt']).dt.month
        df['day'] = pd.to_datetime(df['dt']).dt.day


        # 处理天气缺失值&异常值
        ## step1: 用空值替换气温异常值
        df.loc[(df['day_temperature'] > 100) | (df['day_temperature'] < -100), 'day_temperature'] = np.nan
        df.loc[(df['night_temperature'] > 100) | (df['night_temperature'] < -100), 'night_temperature'] = np.nan

        ## step2: 用月份平均气温填补空值
        month_dtemp = df.groupby('month', as_index=False)['day_temperature'].agg({'month_dtemp': np.mean})
        month_ntemp = df.groupby('month', as_index=False)['night_temperature'].agg({'month_ntemp': np.mean})
        df = df.merge(month_dtemp, how='left', on=['month'])
        df = df.merge(month_ntemp, how='left', on=['month'])
        day_nan = df[df['day_temperature'].isnull() == True].index
        night_nan = df[df['night_temperature'].isnull() == True].index
        df.loc[day_nan, 'day_temperature'] = df.loc[day_nan, 'month_dtemp']
        df.loc[night_nan, 'night_temperature'] = df.loc[night_nan, 'month_ntemp']

        ## step3: 整体填补&去掉冗余列
        df[['day_temperature', 'night_temperature']] \
            = df[['day_temperature', 'night_temperature']].fillna(method='bfill') \
            .fillna(method='ffill') \
            .fillna(0.)
        df = df.drop(['month_dtemp', 'month_ntemp'], axis=1)


        # 数据截取&空值处理
        if is_train:
            df = df[df.dt < pd.to_datetime(self.forecast_date)]
            df = df.dropna()
        else:
            df = df[df.dt >= pd.to_datetime(self.forecast_date)]
        return df

