#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------
# File  : cat1_common_data_process.py
# Author: liushichang
# Date  : 2021/3/23
# Desc  : 品类模型数据处理类
# Contact : liushichang@meituan.com
# ----------------------------------

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class SaleDataPreprocess(object):

    def __init__(self, cat1_id, forecast_date):
        self.cat1_id = cat1_id
        self.forecast_date = forecast_date

    @staticmethod
    def raw_data(df_train_raw, df_pred_raw):
        """
        加载原始数据
        """
        # 训练数据
        df_train_raw['date'] = df_train_raw['dt'].apply(lambda x: x.strftime('%Y%m%d'))  # 设置冗余列‘date’
        df_train_raw = df_train_raw.sort_values('date').reset_index(drop=True)
        # 预测数据
        df_pred_raw['date'] = df_pred_raw['dt'].apply(lambda x: x.strftime('%Y%m%d'))  # 设置冗余列‘date’
        df_pred_raw = df_pred_raw.sort_values('date').reset_index(drop=True)
        return df_train_raw, df_pred_raw

    @staticmethod
    def get_longtail_data(data):
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

    @staticmethod
    def drop_Outliers(train_data, top, bottom):
        """
        滑动过滤异常值
        :param train_data: 训练数据
        :param top: 过滤上限
        :param bottom: 过滤下限
        :return: 过滤后的数据
        """
        dropped_index = []  # 要过滤的数据索引

        for sku in train_data.sku_id.unique():
            df_sku = train_data[train_data.sku_id == sku]

            # 区分促销/非促销数据
            # 非促销日期数据
            data_normal = df_sku[(df_sku['pro_num'] == 0) &
                                 (df_sku['seq_num'] == 0) &
                                 (df_sku['csu_redu_num'] == 0)]['arranged_cnt']

            # 促销日期数据
            data_pro = df_sku[(df_sku['pro_num'] != 0) |
                              (df_sku['seq_num'] != 0) |
                              (df_sku['csu_redu_num'] != 0)]['arranged_cnt']

            # 滑动窗口获取异常值索引
            # 非促销日异常值
            if len(data_normal) >= 7:
                for i in range(7, len(data_normal) + 1):
                    window = data_normal.iloc[i - 7:i]
                    avg = window.mean()
                    dropped_index.extend(window[(window < avg * bottom) | (window > avg * top)].index)
            else:
                avg = data_normal.mean()
                dropped_index.extend(data_normal[(data_normal < avg * bottom) | (data_normal > avg * top)].index)
            # 促销日异常值
            if len(data_pro) >= 7:
                for j in range(7, len(data_pro) + 1):
                    window = data_pro.iloc[j - 7:j]
                    avg = window.mean()
                    dropped_index.extend(window[(window < avg * bottom) | (window > avg * top)].index)
            else:
                avg = data_pro.mean()
                dropped_index.extend(data_pro[(data_pro < avg * bottom) | (data_pro > avg * top)].index)

        return train_data.drop(index=dropped_index, axis=0)

    @staticmethod
    def align_train_n_test_wh(df_train_raw, df_pred_raw):
        """
        对齐训练集与测试集的仓库
        Args:
            df_train_raw: 原始训练集数据
            df_pred_raw: 原始测试集数据

        Returns:
            对齐后的数据
        """
        common_wh = set(df_train_raw.wh_id.unique()) & set(df_pred_raw.wh_id.unique())
        df_train_raw = df_train_raw[df_train_raw.wh_id.isin(common_wh)]
        df_pred_raw = df_pred_raw[df_pred_raw.wh_id.isin(common_wh)]
        return df_train_raw, df_pred_raw

    @staticmethod
    def fix_price(df_train_total, pred_raw):
        """
        修正特征中的价格
        Args:
            df_train_total:
            pred_raw:

        Returns:
            修正后数据
        """
        for sku, sku_data in df_train_total.groupby('sku_id'):
            last_price = sku_data['csu_origin_price'].iloc[-1]
            if sku not in pred_raw.sku_id.unique():
                continue
            test_price = pred_raw[pred_raw.sku_id == sku]['csu_origin_price'].unique()[0]
            if test_price != last_price:
                pred_raw.loc[pred_raw.sku_id == sku, 'csu_origin_price'] = last_price
        return df_train_total, pred_raw

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
        # 填补税率&原价缺失值
        for sku_id, data in df.groupby('sku_id'):
            nonan_data.append(data[['tax_rate', 'csu_origin_price']] \
                              .fillna(method='bfill') \
                              .fillna(method='ffill') \
                              .fillna(0.))

        # 替换税率&原价缺失值
        df_nonan = pd.concat(nonan_data).reindex(index=df.index)
        df[['tax_rate', 'csu_origin_price']] = df_nonan

        # 填补加权价格缺失值
        if self.cat1_id in [10021228]:
            if not is_train:
                df['w_price'] = df[['csu_origin_price', 'seq_price', 'discount_price']].min(axis=1)
            else:
                df.loc[df['w_price'].isnull() == True, 'w_price'] \
                    = df.loc[df['w_price'].isnull() == True, ['csu_origin_price', 'seq_price', 'discount_price']] \
                    .min(axis=1)

        # 处理节假日缺失值
        df['festival_name'].fillna('无', inplace=True)
        df['western_festival_name'].fillna('无', inplace=True)

        # 填补促销数据缺失值
        df['seq_num'].fillna(0, inplace=True)
        df['csu_redu_num'].fillna(0, inplace=True)
        df['cir_redu_num'].fillna(0, inplace=True)
        df['pro_num'].fillna(0, inplace=True)

        if self.cat1_id in [10021228]:
            # 用加权价格填补促销价
            df.loc[(df['seq_price'].isnull() == True) & (df['seq_num'] > 0), 'seq_price'] \
                = df.loc[(df['seq_price'].isnull() == True) & (df['seq_num'] > 0), 'w_price']
            df.loc[(df['discount_price'].isnull() == True) & (df['pro_num'] > 0), 'discount_price'] \
                = df.loc[(df['discount_price'].isnull() == True) & (df['pro_num'] > 0), 'w_price']

        # 用原价填补(剩余)缺失的促销价格
        mprice_nan = df[df['seq_price'].isnull() == True].index
        dprice_nan = df[df['discount_price'].isnull() == True].index
        df.loc[mprice_nan, 'seq_price'] = df.loc[mprice_nan, 'csu_origin_price']
        df.loc[dprice_nan, 'discount_price'] = df.loc[dprice_nan, 'csu_origin_price']

        # 价格异常数据修正
        # 异常1: 售价<促销价
        sprice_odd_list = df[df['csu_origin_price'] < df['seq_price']].index.tolist()
        pprice_odd_list = df[df['csu_origin_price'] < df['discount_price']].index.tolist()
        df.loc[sprice_odd_list, 'seq_price'] = df.loc[sprice_odd_list, 'csu_origin_price']
        df.loc[sprice_odd_list, 'seq_num'] = 0
        df.loc[pprice_odd_list, 'discount_price'] = df.loc[pprice_odd_list, 'csu_origin_price']
        df.loc[pprice_odd_list, 'pro_num'] = 0

        # 异常2: (售价=促销价)&促销场次数>0
        pnum_odd_list = df[(df['csu_origin_price'] == df['discount_price']) &
                           (df['pro_num'] > 0)].index.tolist()
        snum_odd_list = df[(df['csu_origin_price'] == df['seq_price']) &
                           (df['seq_num'] > 0)].index.tolist()
        df.loc[pnum_odd_list, 'pro_num'] = 0
        df.loc[snum_odd_list, 'seq_num'] = 0

        # 处理天气缺失值&异常值
        # step1: 用空值替换气温异常值
        df.loc[(df['avg_t'] > 100) | (df['avg_t'] < -100), 'avg_t'] = np.nan
        df.loc[(df['avg_t_b'] > 100) | (df['avg_t_b'] < -100), 'avg_t_b'] = np.nan

        # step2: 用月份平均气温填补空值
        df['month'] = pd.to_datetime(df['dt']).dt.month
        month_temp = df.groupby('month', as_index=False)['avg_t'].agg({'month_temp': np.mean})
        df = df.merge(month_temp, how='left', on=['month'])
        t_nan = df[df['avg_t'].isnull() == True].index
        tb_nan = df[df['avg_t_b'].isnull() == True].index
        df.loc[t_nan, 'avg_t'] = df.loc[t_nan, 'month_temp']
        df.loc[tb_nan, 'avg_t_b'] = df.loc[tb_nan, 'month_temp']

        # step3: 整体填补&去掉冗余列
        df[['avg_t', 'avg_t_b']] = df[['avg_t', 'avg_t_b']].fillna(method='bfill') \
            .fillna(method='ffill') \
            .fillna(0.)
        df = df.iloc[:, :-2]

        # 填补天气缺失值
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

    def filter_dropped_cols(self, df_train_raw, df_pred_raw):
        """
        过滤不需要的特征及不满20天的测试数据
        Args:
            df_train_raw: 原始训练特征
            df_pred_raw: 原始预测特征

        Returns:
            过滤后的数据
        """
        # 依据品类过滤不需要的特征字段
        if self.cat1_id in [10021202]:
            dropped_features = ['w_price', 'redu_num']
        else:
            dropped_features = ['redu_num']
        for feature in dropped_features:
            df_train_raw.drop(feature, axis=1, inplace=True)
            df_pred_raw.drop(feature, axis=1, inplace=True)

        total_wh_data = []
        for wh_id, wh_data in df_pred_raw.groupby('wh_id'):
            # 过滤测试数据不足20天的sku
            sku_size = wh_data.groupby('sku_id').size()
            less_20d_skus = sku_size[sku_size < 20].index
            wh_data = wh_data[~wh_data.sku_id.isin(less_20d_skus)]
            total_wh_data.append(wh_data)
        df_pred_raw = pd.concat(total_wh_data)
        return df_train_raw, df_pred_raw

    def pro_outstocks(self, df_train_raw):
        """
        缺货日期销量修正
        Args:
            df_train_raw: 训练数据

        Returns:
            修正后的训练数据
        """
        # 鲜蛋直接剔除缺货数据
        if self.cat1_id in [10021228]:
            df_train_raw = df_train_raw[~((df_train_raw.is_outstock == 1) & (df_train_raw.arranged_cnt == 0))]
            return df_train_raw
        outstocks = df_train_raw.loc[df_train_raw.is_outstock == 1]
        part_data = df_train_raw[(df_train_raw.wh_id.isin(outstocks.wh_id.unique())) &
                                 (df_train_raw.sku_id.isin(outstocks.sku_id.unique()))]

        # 历史3天/历史7天销量
        avg_data = []
        for sku, sku_data in part_data.groupby(['wh_id', 'sku_id']):
            roll_avg_7d = sku_data['arranged_cnt'].rolling(7, min_periods=1).mean()
            roll_avg_3d = sku_data['arranged_cnt'].rolling(3, min_periods=1).mean()
            roll_avg_7d = roll_avg_7d.shift(1).fillna(method='bfill')
            roll_avg_3d = roll_avg_3d.shift(1).fillna(method='bfill')
            avg_data.append(pd.DataFrame({'dt': sku_data.dt,
                                          'wh_id': sku_data.wh_id,
                                          'sku_id': sku_data.sku_id,
                                          'roll_avg_7d': roll_avg_7d,
                                          'roll_avg_3d': roll_avg_3d}))

        avg_data = pd.concat(avg_data)
        df_train_raw = df_train_raw.merge(avg_data, how='left', on=['dt', 'wh_id', 'sku_id'])

        # 取最大值更正销量
        df_train_raw.loc[df_train_raw.is_outstock == 1, 'arranged_cnt'] \
            = df_train_raw.loc[df_train_raw.is_outstock == 1, ['roll_avg_7d', 'roll_avg_3d', 'arranged_cnt']]\
            .max(axis=1)
        df_train_raw.drop(['roll_avg_7d', 'roll_avg_3d'], axis=1, inplace=True)

        return df_train_raw.sort_values('dt')

    def fix_temperature(self, train_raw, pred_raw):
        """
        修正异常温度
        Args:
            train_raw:
            pred_raw:

        Returns:
            温度修正后数据
        """
        train_last_dt = train_raw.sort_values('dt').dt.iloc[-1]
        train_last_temp = train_raw.sort_values('dt').avg_t.iloc[-1]
        if train_last_dt == pd.to_datetime(self.forecast_date) - timedelta(1):
            if not np.isnan(train_last_temp):
                pred_raw.loc[pred_raw.dt == pd.to_datetime(self.forecast_date), 'avg_t_b'] = train_last_temp
        return train_raw, pred_raw
