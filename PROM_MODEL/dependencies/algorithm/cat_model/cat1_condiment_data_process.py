#!/bin/usr/env python3.6
# -*- coding: utf-8 -*-
# ---------------------------------
# ProjectName: data-forecast-spark
# Description: 干调品类模型数据预处理
# Author: songzhen07
# CreateTime: 2021-02-23 17:33
# ---------------------------------

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')


class SaleDataPreprocess:

    def __init__(self, forecast_date):
        self.forecast_date = forecast_date


    def pro_outstocks(self, train_raw):
        """
        缺货日期销量修正
        """
        outstocks = train_raw.loc[train_raw.is_outstock == 1]
        part_data = train_raw[train_raw.sku_id.isin(outstocks.sku_id.unique())]

        # 计算历史7天销量均值
        avg_data = []
        for sku, sku_data in part_data.groupby('sku_id'):
            roll_avg_7d = sku_data['arranged_cnt'].rolling(7, min_periods=1).mean()
            roll_avg_7d = roll_avg_7d.shift(1).fillna(method='bfill')
            avg_data.append(pd.DataFrame({'dt': sku_data.dt,
                                          'bu_id': sku_data.bu_id,
                                          'wh_id': sku_data.wh_id,
                                          'sku_id': sku_data.sku_id,
                                          'roll_avg_7d': roll_avg_7d}))

        avg_data = pd.concat(avg_data)
        train_raw = train_raw.merge(avg_data, how='left', on=['dt', 'bu_id', 'wh_id', 'sku_id'])

        # 取较大值更正销量
        train_raw.loc[train_raw.is_outstock == 1, 'arranged_cnt'] \
            = train_raw.loc[train_raw.is_outstock == 1, ['roll_avg_7d', 'arranged_cnt']] \
            .max(axis=1)
        train_raw.drop(['roll_avg_7d'], axis=1, inplace=True)
        return train_raw.sort_values('dt')


    def sku_to_train(self, data):
        """
        近30天有售卖记录的商品
        :param data: 原始数据
        :return: 商品列表
        """
        b_1d_fc_dt = pd.to_datetime(self.forecast_date) - timedelta(1)
        b_30d_fc_dt = pd.to_datetime(self.forecast_date) - timedelta(30)

        sku_30d_arr = data[(data['dt'] <= b_1d_fc_dt) & (data['dt'] >= b_30d_fc_dt)]
        sku_30d_arr_sum = sku_30d_arr.groupby('sku_id', as_index=False)[['arranged_cnt']].sum()
        sku_train_list = sku_30d_arr_sum[sku_30d_arr_sum['arranged_cnt'] != 0]['sku_id'].tolist()

        return sku_train_list


    def get_longtail_data(self, data):
        """
        获取长尾sku
        :param data: 原始训练集
        :return: 长尾sku数据
        """
        longtails = []
        for sku, sku_data in data.groupby('sku_id'):
            sale = sku_data['arranged_cnt']
            bu_id = sku_data['bu_id'].unique()[0]
            wh_id = sku_data['wh_id'].unique()[0]
            cat2_name = sku_data['cat2_name'].unique()[0]
            if len(sale) >= 7:
                roll_sale = sale.rolling(7, min_periods=7).sum()
                avg_sale = roll_sale.mean()
                if avg_sale < 7:
                    longtails.append([bu_id, wh_id, sku, cat2_name, 7, avg_sale])
            else:
                if sale.sum() < len(sale):
                    longtails.append([bu_id, wh_id, sku, cat2_name, len(sale), sale.sum()])
        longtail_data = pd.DataFrame(longtails,
                                     columns=['bu_id', 'wh_id', 'sku_id', 'cat2_name', 'sale_days', 'sale_avg'])
        return longtail_data


    def holiday_label_process(self, data, dt_field):
        """
        日期特征处理
        """
        #构造日期特征
        data['year'] = pd.to_datetime(data[dt_field]).dt.year
        data['month'] = pd.to_datetime(data[dt_field]).dt.month
        data['day'] = pd.to_datetime(data[dt_field]).dt.day

        #重叠节假日修正
        fetvls = set(data['western_festival_name'].unique()) | set(data['festival_name'].unique())
        ##重叠节日
        multi_fetvl = []
        for f in fetvls:
            if ',' in f:
                multi_fetvl.append(f)
        if len(multi_fetvl) > 0:
            ##截取第一个节日标记
            for mf in multi_fetvl:
                if mf in data['festival_name'].unique():
                    data.loc[(data['festival_name'] == mf), 'festival_name'] = mf[0:mf.rfind(',')]
                else:
                    data.loc[(data['western_festival_name'] == mf), 'western_festival_name'] = mf[0:mf.rfind(',')]

        #根据业务特点修改部分日期标记
        ##受国庆影响,9.30和10.1当日销量下滑明显,将上述日期做特殊标记
        data.loc[(data.month == 9) & (data.day == 30), 'festival_name'] = '国庆前'
        data.loc[(data.month == 10) & (data.day == 1), 'festival_name'] = '国庆当日'
        ## 元旦前一天日期标记
        data.loc[(data.month == 12) & (data.day == 31), 'festival_name'] = '元旦前'
        return data


    def preprocess(self, df_raw, is_train=True):
        """
        数据预处理
        :param df_raw: 原始数据
        :param is_train: 训练集标识
        :return: 处理后的数据
        """
        df = df_raw.sort_values('dt')
        nonan_data = []

        # 处理brand特征
        df['brand_name'].fillna('无', inplace=True)


        # 处理税率&原价&加权价格缺失值
        ## 填补税率&原价缺失值
        for sku_id, data in df.groupby('sku_id'):
            nonan_data.append(data[['tax_rate', 'csu_origin_price']] \
                              .fillna(method='bfill') \
                              .fillna(method='ffill') \
                              .fillna(0.))

        ## 替换税率&原价缺失值
        df_nonan = pd.concat(nonan_data).reindex(index=df.index)
        df[['tax_rate', 'csu_origin_price']] = df_nonan


        # 日期处理
        df['festival_name'].fillna('无', inplace=True)
        df['western_festival_name'].fillna('无', inplace=True)
        df = self.holiday_label_process(df, 'dt')


        # 填补促销数据缺失值
        promo_cols = ['seq_num', 'csu_redu_num', 'cir_redu_num', 'pro_num',
                      'cir_redu_adct', 'cir_redu_mdct', 'csu_redu_adct', 'csu_redu_mdct']
        for col in promo_cols:
            df[col].fillna(0, inplace=True)

        ## 用原价填补(剩余)缺失的促销价格
        mprice_nan = df[df['seq_price'].isnull() == True].index
        dprice_nan = df[df['discount_price'].isnull() == True].index
        df.loc[mprice_nan, 'seq_price'] = df.loc[mprice_nan, 'csu_origin_price']
        df.loc[dprice_nan, 'discount_price'] = df.loc[dprice_nan, 'csu_origin_price']


        # 价格异常数据修正
        ## 异常1: 售价<促销价
        sprice_odd_list = df[df['csu_origin_price'] < df['seq_price']].index.tolist()
        pprice_odd_list = df[df['csu_origin_price'] < df['discount_price']].index.tolist()
        df.loc[sprice_odd_list, 'seq_price'] = df.loc[sprice_odd_list, 'csu_origin_price']
        df.loc[sprice_odd_list, 'seq_num'] = 0
        df.loc[pprice_odd_list, 'discount_price'] = df.loc[pprice_odd_list, 'csu_origin_price']
        df.loc[pprice_odd_list, 'pro_num'] = 0

        ## 异常2: (售价=促销价)&促销场次数>0
        pnum_odd_list = df[(df['csu_origin_price'] == df['discount_price']) &
                           (df['pro_num'] > 0)].index.tolist()
        snum_odd_list = df[(df['csu_origin_price'] == df['seq_price']) &
                           (df['seq_num'] > 0)].index.tolist()
        df.loc[pnum_odd_list, 'pro_num'] = 0
        df.loc[snum_odd_list, 'seq_num'] = 0


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
        df[['avg_t', 'avg_t_b']] = df[['avg_t', 'avg_t_b']] \
            .fillna(method='bfill') \
            .fillna(method='ffill') \
            .fillna(0.)
        df = df.drop('month_temp', axis=1)

        ## 填补天气缺失值
        df['weather'].fillna('晴', inplace=True)
        df['weather_b'].fillna('晴', inplace=True)


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