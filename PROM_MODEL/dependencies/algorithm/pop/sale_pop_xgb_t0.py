#!/bin/usr/env python3.6
# -*- coding: utf-8 -*-
# ---------------------------------
# ProjectName: data-forecast-server
# Description: 
# Author: liushichang
# CreateTime: 2021-05-19 11:20
# ---------------------------------
import gc
import multiprocessing
import os
import random
import time
import traceback
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBRegressor

from dependencies.common.forecast_utils import log_print, reduce_mem_usage

random.seed(42)
np.random.seed(42)

warnings.filterwarnings('ignore')


class SalePopXgbForecast:

    def __init__(self, forecast_date):
        self.forecast_date = forecast_date  # 预测日期
        self.raw_data = None
        self.train_data = None
        self.longtail_data = None
        self.lr_train = None
        self.lr_test = None
        self.df_test = None

    def sku_to_train(self, df):
        """
        返回近30天有售卖的sku
        :param df: 训练集数据
        :return: 近30日有售卖记录SKU
        """
        sku_list = []
        for key, data in df.groupby(['wh_id', 'bu_id']):
            b_1d_fc_dt = pd.to_datetime(self.forecast_date) - timedelta(1)
            b_30d_fc_dt = pd.to_datetime(self.forecast_date) - timedelta(30)

            sku_30d_arr = data[(data['dt'] <= b_1d_fc_dt) & (data['dt'] >= b_30d_fc_dt)]
            sku_30d_arr_sum = sku_30d_arr.groupby('sku_id', as_index=False)[['arranged_cnt']].sum()
            sku_col = sku_30d_arr_sum[sku_30d_arr_sum['arranged_cnt'] != 0]['sku_id'].tolist()

            wh_col = [key[0]] * len(sku_col)
            bu_col = [key[1]] * len(sku_col)

            sku_list.append(pd.DataFrame({'wh_id': wh_col,
                                          'bu_id': bu_col,
                                          'sku_id': sku_col}))
        return pd.concat(sku_list)

    def get_longtail_data(self, data):
        """
        获取长尾SKU
        :param data: 训练集数据
        :return: 长尾SKU
        """
        longtails = []
        for key, sku_data in data.groupby(['wh_id', 'bu_id', 'sku_id']):
            sale = sku_data['arranged_cnt']
            wh_id = key[0]
            bu_id = key[1]
            sku_id = key[2]
            cat2_name = sku_data['cat2_name'].unique()[0]
            if len(sale) >= 7:
                roll_sale = sale.rolling(7, min_periods=7).sum()
                avg_sale = roll_sale.mean()
                if avg_sale < 7:
                    longtails.append([wh_id, bu_id, sku_id, cat2_name, 7, avg_sale])
            else:
                if sale.sum() < len(sale):
                    longtails.append([wh_id, bu_id, sku_id, cat2_name, len(sale), sale.sum()])
        longtail_data = pd.DataFrame(longtails,
                                     columns=['wh_id', 'bu_id', 'sku_id', 'cat2_name', 'sale_days', 'sale_avg'])
        return longtail_data

    def prepro_func(self, df_raw, is_train=True):
        """
        特征预处理函数
        :param is_train:
        :return:
        """
        df = df_raw.sort_values('dt')
        nonan_data = []

        # 处理brand特征
        df['brand_name'].fillna('无', inplace=True)

        # 原价缺失值
        for key, data in df.groupby(['wh_id', 'bu_id', 'sku_id']):
            nonan_data.append(data[['origin_price']] \
                              .fillna(method='bfill') \
                              .fillna(method='ffill'))
        df_nonan = pd.concat(nonan_data).reindex(index=df.index)
        df[['origin_price']] = df_nonan

        # 用原价填补(剩余)缺失的促销价格
        mprice_nan = df[df['seq_price'].isnull() == True].index
        df.loc[mprice_nan, 'seq_price'] = df.loc[mprice_nan, 'origin_price']

        # 原价/促销价数据修正
        # 售价低于促销价
        sprice_odd_list = df[df['origin_price'] < df['seq_price']].index.tolist()
        df.loc[sprice_odd_list, 'seq_price'] = df.loc[sprice_odd_list, 'origin_price']

        # 处理天气缺失值&异常值
        # step1: 用空值替换气温异常值
        df.loc[(df['day_temperature'] > 100) | (df['day_temperature'] < -100), 'day_temperature'] = np.nan
        df.loc[(df['night_temperature'] > 100) | (df['night_temperature'] < -100), 'night_temperature'] = np.nan

        # step2: 用月份平均气温填补空值
        month_temp = df.groupby(['wh_id', 'month'], as_index=False)['day_temperature'].agg({'month_temp': np.mean})
        df = df.merge(month_temp, how='left', on=['wh_id', 'month'])
        t_nan = df[df['day_temperature'].isnull() == True].index
        tb_nan = df[df['night_temperature'].isnull() == True].index
        df.loc[t_nan, 'day_temperature'] = df.loc[t_nan, 'month_temp']
        df.loc[tb_nan, 'night_temperature'] = df.loc[tb_nan, 'month_temp']

        # step3: 整体填补&去掉冗余列
        df[['day_temperature', 'night_temperature']] \
            = df[['day_temperature', 'night_temperature']].fillna(method='bfill') \
            .fillna(method='ffill') \
            .fillna(0.)
        df = df.drop('month_temp', axis=1)

        # 数据截取&空值处理
        if is_train:
            df = df[df.dt < pd.to_datetime(self.forecast_date)]
            df = df.dropna()
        else:
            df = df[df.dt >= pd.to_datetime(self.forecast_date)]
        return df

    def choose_data(self, part, data):
        """
        构造LR输入矩阵
        :param data: 原始训练集
        :return: 历史销量矩阵
        """
        if len(data) == 0:
            return pd.DataFrame([])
        final_rst = []
        for key, sku_data in data.groupby(['bu_id', 'sku_id']):
            y_val = sku_data['arranged_cnt']

            df_sku = pd.DataFrame(y_val)
            cols, names = list(), list()

            ## 生产t-14~t-1日销量特征
            for i in range(14, 0, -1):
                cols.append(df_sku.shift(i))
                names.append('cnt(t-%d)' % i)

            ## 生产t0日数据
            cols.append(df_sku)
            names.append('cnt(t-0)')

            df_cnt = pd.concat(cols, axis=1)
            df_cnt.columns = names

            ## 过滤空值
            df_cnt.dropna(inplace=True)

            ## 保留全部结果
            df_saved = df_cnt.copy()
            df_ids = pd.DataFrame({'bu_id': [key[0]] * len(df_saved),
                                   'sku_id': [key[1]] * len(df_saved)}, index=df_saved.index)
            df_rst = pd.concat([df_saved, df_ids], axis=1)
            final_rst.append(df_rst)

        try:
            return pd.concat(final_rst)
        except Exception as e:
            traceback.print_exc()
            return pd.DataFrame([])

    def multi_process(self, wh_id, raw_data, nthreads=1):
        """
        多进程处理函数
        :param raw_data: 原始数据
        :param nthreads: 进程数量
        :return: 处理后数据
        """
        log_print("{}仓 正在开启 输入矩阵构造 多进程, 父进程Id: {}".format(str(wh_id), str(os.getpid())))

        # 创建子进程
        process_num = nthreads
        pool = multiprocessing.Pool(process_num)

        # 存放子进程返回结果
        mid_results = []

        # 将sku打乱&分组
        bu_array = raw_data.bu_id.unique()
        np.random.shuffle(bu_array)
        bu_group = np.array_split(bu_array, process_num, axis=0)

        # 调用子进程处理
        for part, bus in enumerate(bu_group):
            part_train = raw_data[raw_data.bu_id.isin(bus)]
            # 调用处理函数
            mid_results.append(pool.apply_async(self.build_lag_feats, args=(part_train,)))
        pool.close()
        pool.join()
        log_print("{}仓 输入矩阵构造 多进程已结束.".format(str(wh_id)))

        # 聚合得到最终结果
        final_data = []
        for part_result in mid_results:
            final_data.append(part_result.get())
        return pd.concat(final_data)

    def lr_data_prepare(self, df):
        """
        LR训练数据准备
        :return: 模型训练数据
        """
        log_print("开始预处理数据...")
        begin_time = time.time()
        # 读取数据
        raw_data = df

        # 只保留近30天有售卖的SKU
        trained_sku = self.sku_to_train(raw_data[raw_data.is_train == 1])
        cleared_data = raw_data.merge(trained_sku, how='inner', on=trained_sku.columns.tolist())
        log_print("完成保留近30天有售卖的SKU.")

        # 日期处理
        cleared_data = self.holiday_label_process(cleared_data, dt_field='dt')

        # 预处理数据
        train_data = self.prepro_func(cleared_data[cleared_data.is_train == 1], is_train=True)
        test_data = self.prepro_func(cleared_data[cleared_data.is_train == 0], is_train=False)
        log_print("完成特征预处理.")
        # 长尾SKU过滤
        ## 获取长尾数据
        longtail_data = self.get_longtail_data(train_data)
        ## (从训练集中)过滤长尾sku
        temp = train_data.merge(longtail_data[['wh_id', 'bu_id', 'sku_id', 'sale_avg']],
                                how='left', on=['wh_id', 'bu_id', 'sku_id'])
        train_data = temp[temp['sale_avg'].isnull() == True].drop('sale_avg', axis=1)
        log_print("训练数据中过滤掉长尾数据.")

        # 保留训练&测试集公共SKU
        cols = ['wh_id', 'bu_id', 'sku_id']
        df_common = train_data[cols].merge(test_data[cols], how='inner', on=cols) \
            .drop_duplicates(subset=cols, keep='first')

        train_data = train_data.merge(df_common, how='inner', on=cols)
        test_data = test_data.merge(df_common, how='inner', on=cols)
        log_print("保留训练集与测试集同时出现的sku数据.")

        # 特征编码
        ## 将事业部和仓合并后统一编码
        total_data = pd.concat([train_data, test_data])
        total_data['discount'] = (1 - total_data['seq_price'] / total_data['origin_price']) * 100

        map_fields = ['cat1_name', 'cat2_name', 'day_abbr', 'festival_name']
        total_data[map_fields] = total_data[map_fields].astype('category')

        total_data[map_fields] = total_data[map_fields].apply(lambda x: x.cat.codes)
        total_data.set_index('dt', inplace=True)
        log_print("数据预处理完成, 用时: {}s".format(str(time.time() - begin_time)))

        # 筛选训练数据
        log_print("正在构造LR输入矩阵...")
        begin_time = time.time()
        total_cnt_data = []

        for wh_id, _df in total_data.groupby('wh_id'):
            cnt_data = self.multi_process(wh_id, _df, nthreads=2)
            cnt_data['wh_id'] = wh_id
            total_cnt_data.append(cnt_data)

        log_print("LR输入矩阵构造完成，用时: {}s".format(str(time.time() - begin_time)))

        # 特征筛选
        df_total = pd.concat(total_cnt_data)
        df_total = df_total.reset_index().rename(columns={'index': 'dt'})

        features = ['wh_id', 'bu_id', 'sku_id', 'cat1_name', 'cat2_name', 'discount',
                    'day_abbr', 'is_work_day', 'is_weekend', 'is_holiday', 'festival_name',
                    'day_temperature', 'night_temperature', 'month']
        key_cols = ['wh_id', 'bu_id', 'sku_id', 'dt']
        df_features = total_data[features].reset_index().rename(columns={'index': 'dt'})

        df_total = df_total.merge(df_features, how='inner', on=['wh_id', 'bu_id', 'sku_id', 'dt'])

        ## 特征排序
        left_features = list(set(features) - set(key_cols))
        ordered_cols = ['cnt(t-%d)' % i for i in range(14, 0, -1)] + left_features + ['cnt(t-0)']

        df_train = df_total[df_total.dt != pd.to_datetime(self.forecast_date)]
        df_test = df_total[df_total.dt == pd.to_datetime(self.forecast_date)]

        self.lr_train = df_train[ordered_cols].values
        self.lr_test = df_test[ordered_cols].values
        self.train_data = train_data
        self.longtail_data = longtail_data
        self.raw_data = raw_data
        self.df_test = df_test

        log_print("LR模型训练数据就绪.")
        return

    def longtail_pre(self, df):
        """
        长尾预测逻辑
        :param df: 长尾数据
        :return: 长尾预测结果
        """
        data = df.sort_values('dt')
        fc_dt = pd.to_datetime(self.forecast_date)
        result_list = []

        for key, values in data.groupby(['wh_id', 'bu_id', 'sku_id']):
            avg_val = values.avg_result.iloc[0]
            cat1_name = values.cat1_name.iloc[0]
            data = values.arranged_cnt

            if cat1_name == '蔬菜水果':
                if data.iloc[-1] <= 1:
                    pred_uc = data.iloc[-1]
                else:
                    pred_uc = data.iloc[-1] + np.random.normal(0, 0.1)

                if pred_uc < 0:
                    pred_uc = data.iloc[-1]
            else:
                pred_uc = avg_val

            _result = pd.DataFrame({'wh_id': [key[0]],
                                    'bu_id': [key[1]],
                                    'sku_id': [key[2]],
                                    'prediction': [pred_uc],
                                    'dt': [fc_dt]
                                    })
            result_list.append(_result)

        return pd.concat(result_list)

    def add_cnt_band(self, train_df):
        cnt_df = train_df[train_df.dt > (pd.to_datetime(self.forecast_date) - pd.Timedelta("30 days"))]
        cnt_df = cnt_df.sort_values('dt', ascending=False).groupby(['sku_id', 'wh_id', 'bu_id']).head(7)\
            .groupby(['sku_id', 'wh_id', 'bu_id'])['arranged_cnt'].mean().reset_index()
        cnt_df['cnt_band'] = cnt_df['arranged_cnt'] // 5
        cnt_df.drop('arranged_cnt', axis=1, inplace=True)
        return cnt_df

    def preprocess_data(self, raw_rdd, config):
        log_print("开始预处理数据...")
        begin_time = time.time()
        raw_data = pd.DataFrame(raw_rdd, columns=config['schema_cols']).replace('', np.nan)
        raw_data['dt'] = pd.to_datetime(raw_data['dt'], format='%Y%m%d')

        try:
            # 添加日期计数
            raw_data = self.add_seq_day_gap(raw_data)

            # 只保留近30天有售卖的SKU
            trained_sku = self.sku_to_train(raw_data[raw_data.is_train == 1])
            cleared_data = raw_data.merge(trained_sku, how='inner', on=trained_sku.columns.tolist())
            log_print("完成保留近30天有售卖的SKU.")

            del trained_sku
            gc.collect()

            # 日期处理
            cleared_data = SalePopXgbForecast.holiday_label_process(cleared_data, dt_field='dt')
            # 预处理数据
            train_data = self.prepro_func(cleared_data[cleared_data.is_train == 1], is_train=True)
            test_data = self.prepro_func(cleared_data[cleared_data.is_train == 0], is_train=False)
        except:
            log_print("数据预处理失败，用时: {}s".format(str(time.time() - begin_time)))
            traceback.print_exc()
            return None
        log_print("数据预处理成功，用时: {}s".format(str(time.time() - begin_time)))
        return reduce_mem_usage(pd.concat([train_data, test_data])).to_dict('records')

    def process_longtail_data(self, total_rdd):
        begin_time = time.time()
        raw_data = pd.DataFrame(total_rdd).replace('', np.nan)

        longtail_df = self.get_longtail_data(raw_data[raw_data.is_train == 1])
        log_print("长尾数据预处理成功，用时: {}s".format(str(time.time() - begin_time)))
        return reduce_mem_usage(longtail_df).to_dict('records')

    def generate_features(self, total_rdd, longtail_rdd):
        all_begin_time = time.time()
        total_df = pd.DataFrame(total_rdd).replace('', np.nan)
        longtail_data = pd.DataFrame(longtail_rdd).replace('', np.nan)

        train_data = total_df[total_df.is_train == 1]
        test_data = total_df[total_df.is_train == 0]

        try:
            # (从训练集中)过滤长尾sku
            temp = train_data.merge(longtail_data[['wh_id', 'bu_id', 'sku_id', 'sale_avg']],
                                    how='left', on=['wh_id', 'bu_id', 'sku_id'])
            train_data = temp[temp['sale_avg'].isnull() == True].drop('sale_avg', axis=1)
            log_print("训练数据中过滤掉长尾数据.")

            # 保留训练&测试集公共SKU
            cols = ['wh_id', 'bu_id', 'sku_id']
            df_common = train_data[cols].merge(test_data[cols], how='inner', on=cols) \
                .drop_duplicates(subset=cols, keep='first')

            train_data = train_data.merge(df_common, how='inner', on=cols)
            test_data = test_data.merge(df_common, how='inner', on=cols)
            log_print("保留训练集与测试集同时出现的sku数据.")

            # 特征编码
            # 将事业部和仓合并后统一编码
            total_data = pd.concat([train_data, test_data])
            total_data['discount'] = (1 - total_data['seq_price'] / total_data['origin_price']) * 100

            # 增加销量层级特征
            cnt_df = self.add_cnt_band(train_data)
            total_data = total_data.merge(cnt_df, on=['sku_id', 'wh_id', 'bu_id'], how='left').fillna(0)

            del test_data
            gc.collect()

            # 增加销量平滑特征
            sku_data_df = self.his_sale_avg(total_data)
            total_data = total_data.merge(sku_data_df, on=['sku_id', 'wh_id', 'bu_id', 'dt'])

            del sku_data_df
            gc.collect()

            # 增加 LAG 特征
            log_print("正在构造 LAG 特征...")
            begin_time = time.time()
            total_cnt_data = []
            for wh_id, _df in total_data[['sku_id', 'wh_id', 'bu_id', 'dt', 'arranged_cnt']].groupby('wh_id'):
                cnt_data = self.multi_process(wh_id, _df, nthreads=2)
                cnt_data['wh_id'] = wh_id
                total_cnt_data.append(cnt_data)
            total_cnt_df = pd.concat(total_cnt_data)
            df_total = total_data.merge(total_cnt_df, on=['sku_id', 'wh_id', 'bu_id', 'dt', 'arranged_cnt'])
            log_print("XGB 输入矩阵构造完成，用时: {}s".format(str(time.time() - begin_time)))

            # 特征筛选
            df_train = df_total[df_total.dt != pd.to_datetime(self.forecast_date)]
            df_test = df_total[df_total.dt == pd.to_datetime(self.forecast_date)]
            df_train['dt'] = df_train['dt'].dt.strftime("%Y%m%d")
            df_test['dt'] = df_test['dt'].dt.strftime("%Y%m%d")
        except:
            log_print("特征数据处理失败，用时: {}s".format(str(time.time() - all_begin_time)))
            traceback.print_exc()
            return []

        log_print("特征数据就绪，用时: {}s".format(str(time.time() - all_begin_time)))
        return reduce_mem_usage(pd.concat([df_train, df_test])).to_dict('records')

    def process_features(self, raw_rdd, config):
        """
        特征生产
        Args:
            raw_rdd: 原始数据组成的 RDD
            config: 配置项字典

        Returns:
            处理好后的数据
        """
        log_print("开始预处理数据...")
        begin_time = time.time()
        raw_data = pd.DataFrame(raw_rdd[1], columns=config['schema_cols']).replace('', np.nan)
        raw_data['dt'] = pd.to_datetime(raw_data['dt'], format='%Y%m%d')

        try:
            # 添加日期计数
            raw_data = self.add_seq_day_gap(raw_data)

            # 只保留近30天有售卖的SKU
            trained_sku = self.sku_to_train(raw_data[raw_data.is_train == 1])
            cleared_data = raw_data.merge(trained_sku, how='inner', on=trained_sku.columns.tolist())
            log_print("完成保留近30天有售卖的SKU.")

            del trained_sku
            gc.collect()

            # 日期处理
            cleared_data = SalePopXgbForecast.holiday_label_process(cleared_data, dt_field='dt')
            # 预处理数据
            train_data = self.prepro_func(cleared_data[cleared_data.is_train == 1], is_train=True)
            test_data = self.prepro_func(cleared_data[cleared_data.is_train == 0], is_train=False)
            log_print("完成特征预处理.")

            del cleared_data
            gc.collect()

            # 长尾SKU过滤
            # 获取长尾数据
            longtail_data = self.get_longtail_data(train_data)

            # (从训练集中)过滤长尾sku
            temp = train_data.merge(longtail_data[['wh_id', 'bu_id', 'sku_id', 'sale_avg']],
                                    how='left', on=['wh_id', 'bu_id', 'sku_id'])
            train_data = temp[temp['sale_avg'].isnull() == True].drop('sale_avg', axis=1)
            log_print("训练数据中过滤掉长尾数据.")

            # 保留训练&测试集公共SKU
            cols = ['wh_id', 'bu_id', 'sku_id']
            df_common = train_data[cols].merge(test_data[cols], how='inner', on=cols) \
                .drop_duplicates(subset=cols, keep='first')

            train_data = train_data.merge(df_common, how='inner', on=cols)
            test_data = test_data.merge(df_common, how='inner', on=cols)
            log_print("保留训练集与测试集同时出现的sku数据.")

            # 特征编码
            # 将事业部和仓合并后统一编码
            total_data = pd.concat([train_data, test_data])
            total_data['discount'] = (1 - total_data['seq_price'] / total_data['origin_price']) * 100
            log_print("数据预处理完成, 用时: {}s".format(str(time.time() - begin_time)))

            # 增加销量层级特征
            cnt_df = self.add_cnt_band(train_data)
            total_data = total_data.merge(cnt_df, on=['sku_id', 'wh_id', 'bu_id'], how='left').fillna(0)

            del test_data
            gc.collect()

            # 增加销量平滑特征
            sku_data_df = self.his_sale_avg(total_data)
            total_data = total_data.merge(sku_data_df, on=['sku_id', 'wh_id', 'bu_id', 'dt'])

            del sku_data_df
            gc.collect()

            # 增加 LAG 特征
            log_print("正在构造 LAG 特征...")
            begin_time = time.time()
            total_cnt_data = []
            for wh_id, _df in total_data[['sku_id', 'wh_id', 'bu_id', 'dt', 'arranged_cnt']].groupby('wh_id'):
                cnt_data = self.multi_process(wh_id, _df, nthreads=2)
                cnt_data['wh_id'] = wh_id
                total_cnt_data.append(cnt_data)
            total_cnt_df = pd.concat(total_cnt_data)
            df_total = total_data.merge(total_cnt_df, on=['sku_id', 'wh_id', 'bu_id', 'dt', 'arranged_cnt'])

            log_print("XGB 输入矩阵构造完成，用时: {}s".format(str(time.time() - begin_time)))

            # 特征筛选
            df_train = df_total[df_total.dt != pd.to_datetime(self.forecast_date)]
            df_test = df_total[df_total.dt == pd.to_datetime(self.forecast_date)]
            df_test['dt'] = df_test['dt'].dt.strftime("%Y%m%d")
        except:
            traceback.print_exc()
            return ('df_train', []), ('train_data', []), ('longtail_data', []), ('raw_data', []), ('df_test', [])

        log_print("XGB 模型训练数据就绪.")
        train_data_output = train_data.drop_duplicates(subset=['wh_id', 'bu_id', 'sku_id'], keep='first')[
            config['train_data_cols']]
        longtail_data_output = longtail_data[config['longtail_data_cols']]
        raw_data['dt'] = raw_data['dt'].dt.strftime("%Y%m%d")
        raw_data_output = raw_data[config['raw_data_cols']]
        return ('df_train', reduce_mem_usage(df_train).to_dict('records')), \
               ('train_data', reduce_mem_usage(train_data_output).to_dict('records')), \
               ('longtail_data', reduce_mem_usage(longtail_data_output).to_dict('records')), \
               ('raw_data', reduce_mem_usage(raw_data_output).to_dict('records')), \
               ('df_test', reduce_mem_usage(df_test).to_dict('records'))

    def preprocess_lr_data(self, raw_rdd, config):
        """
        preprocessing data for LR model
        Args:
            raw_rdd: raw data rdd
            config: config dict

        Returns:

        """
        log_print("开始预处理数据...")
        begin_time = time.time()
        raw_data = pd.DataFrame(raw_rdd[1], columns=config['schema_cols']).replace('', np.nan)
        raw_data['dt'] = pd.to_datetime(raw_data['dt'], format='%Y%m%d')

        try:
            # 只保留近30天有售卖的SKU
            trained_sku = self.sku_to_train(raw_data[raw_data.is_train == 1])
            cleared_data = raw_data.merge(trained_sku, how='inner', on=trained_sku.columns.tolist())
            log_print("完成保留近30天有售卖的SKU.")

            # 日期处理
            cleared_data = self.holiday_label_process(cleared_data, dt_field='dt')
            # 预处理数据
            train_data = self.prepro_func(cleared_data[cleared_data.is_train == 1], is_train=True)
            test_data = self.prepro_func(cleared_data[cleared_data.is_train == 0], is_train=False)
            log_print("完成特征预处理.")

            # 长尾SKU过滤
            # 获取长尾数据
            longtail_data = self.get_longtail_data(train_data)

            # (从训练集中)过滤长尾sku
            temp = train_data.merge(longtail_data[['wh_id', 'bu_id', 'sku_id', 'sale_avg']],
                                    how='left', on=['wh_id', 'bu_id', 'sku_id'])
            train_data = temp[temp['sale_avg'].isnull() == True].drop('sale_avg', axis=1)
            log_print("训练数据中过滤掉长尾数据.")

            # 保留训练&测试集公共SKU
            cols = ['wh_id', 'bu_id', 'sku_id']
            df_common = train_data[cols].merge(test_data[cols], how='inner', on=cols) \
                .drop_duplicates(subset=cols, keep='first')

            train_data = train_data.merge(df_common, how='inner', on=cols)
            test_data = test_data.merge(df_common, how='inner', on=cols)
            log_print("保留训练集与测试集同时出现的sku数据.")

            # 特征编码
            # 将事业部和仓合并后统一编码
            total_data = pd.concat([train_data, test_data])
            total_data['discount'] = (1 - total_data['seq_price'] / total_data['origin_price']) * 100
            total_data.set_index('dt', inplace=True)
            log_print("数据预处理完成, 用时: {}s".format(str(time.time() - begin_time)))

            # 筛选训练数据
            log_print("正在构造LR输入矩阵...")
            begin_time = time.time()
            total_cnt_data = []

            for wh_id, _df in total_data.groupby('wh_id'):
                cnt_data = self.multi_process(wh_id, _df, nthreads=2)
                cnt_data['wh_id'] = wh_id
                total_cnt_data.append(cnt_data)

            log_print("LR输入矩阵构造完成，用时: {}s".format(str(time.time() - begin_time)))

            # 特征筛选
            df_total = pd.concat(total_cnt_data)
            df_total = df_total.reset_index().rename(columns={'index': 'dt'})
            df_features = total_data[config['features']].reset_index().rename(columns={'index': 'dt'})

            df_total = df_total.merge(df_features, how='inner', on=['wh_id', 'bu_id', 'sku_id', 'dt'])
            df_train = df_total[df_total.dt != pd.to_datetime(self.forecast_date)]
            df_test = df_total[df_total.dt == pd.to_datetime(self.forecast_date)]
            df_test['dt'] = df_test['dt'].dt.strftime("%Y%m%d")
        except:
            traceback.print_exc()
            return ('df_train', []), ('train_data', []), ('longtail_data', []), ('raw_data', []), ('df_test', [])

        log_print("LR模型训练数据就绪.")
        df_train_output = df_train[config['ordered_cols']]
        df_test_output = df_test[list(set(config['key_cols'] + config['ordered_cols']))]
        train_data_output = train_data.drop_duplicates(subset=['wh_id', 'bu_id', 'sku_id'], keep='first')[
            config['train_data_cols']]
        longtail_data_output = longtail_data[config['longtail_data_cols']]
        raw_data['dt'] = raw_data['dt'].dt.strftime("%Y%m%d")
        raw_data_output = raw_data[config['raw_data_cols']]
        return ('df_train', df_train_output.to_dict('records')), \
               ('train_data', train_data_output.to_dict('records')), \
               ('longtail_data', longtail_data_output.to_dict('records')), \
               ('raw_data', raw_data_output.to_dict('records')), \
               ('df_test', df_test_output.to_dict('records'))


    @staticmethod
    def holiday_label_process(data, dt_field):
        """
        日期特征构造函数
        :param data: 全部数据
        :param dt_field: 日期字段
        :return: 含日期特征的数据
        """
        # 构造日期特征
        data['year'] = pd.to_datetime(data[dt_field]).dt.year
        data['month'] = pd.to_datetime(data[dt_field]).dt.month
        data['day'] = pd.to_datetime(data[dt_field]).dt.day
        data['yr_wk'] = pd.to_datetime(data[dt_field]).dt.year * 100 + pd.to_datetime(data[dt_field]).dt.week

        # 重叠节假日修正
        fetvls = data['festival_name'].unique()
        # 重叠节日
        multi_fetvl = []
        for f in fetvls:
            if ',' in f:
                multi_fetvl.append(f)
        if len(multi_fetvl) > 0:
            # 截取第一个节日标记
            for mf in multi_fetvl:
                data.loc[(data['festival_name'] == mf), 'festival_name'] = mf[0:mf.rfind(',')]

        return data

    @staticmethod
    def get_avg_result(all_data, config):
        """
        历史均值计算
        Args:
            all_data: 原始数据
            config: 配置项字典

        Returns:
            SKU 平均销量数据
        """
        all_data = pd.DataFrame(all_data[1], columns=config['cnt_longtail_cols']).replace('', np.nan)
        avg_rst = []

        for key, group in all_data.groupby(['wh_id', 'bu_id', 'sku_id']):
            cat1_name = group.cat1_name.unique()[0]
            if len(group) < 6:
                sale_avg = group.arranged_cnt.mean()
            else:
                recent_data = group.arranged_cnt.iloc[-6:]
                sale_avg = recent_data.mean()
            avg_rst.append([key[0], key[1], key[2], cat1_name, sale_avg])

        return pd.DataFrame(avg_rst, columns=['wh_id', 'bu_id', 'sku_id', 'cat1_name', 'avg_result']).to_dict('records')

    @staticmethod
    def add_seq_day_gap(df):
        """
        增加日期计数
        Args:
            df: 全部数据

        Returns:
            全部数据
        """
        df['tmp_start_date'] = '2021-01-01'
        df['tmp_start_date'] = pd.to_datetime(df['tmp_start_date'])
        df['seq_day_gap'] = (df['dt'] - df['tmp_start_date']).dt.days
        df.drop('tmp_start_date'.split(','), inplace=True, axis=1)
        return df

    @staticmethod
    def build_lag_feats(tto):
        """
        增加销量 LAG 特征
        Args:
            tto: 按照 ('sku_id', 'bu_id', 'wh_id') 分组后数据

        Returns:
            增加了销量 LAG 特征的数据
        """
        rst_list = []
        for keys, tt in tto.groupby(['sku_id', 'bu_id', 'wh_id']):
            tt.sort_values('dt', inplace=True)
            cur_label_date_dt_list = tt.dt.values
            tt = tt.set_index('dt')
            date_shifts = [1, 7]
            for i in date_shifts:
                tt['shift_arranged_cnt_{}'.format(i)] = tt['arranged_cnt'].shift(i)
                tt['rolling_mean_1_shift_{}'.format(i)] = tt['shift_arranged_cnt_{}'.format(i)].rolling(1).mean()
                tt['rolling_mean_3_shift_{}'.format(i)] = tt['shift_arranged_cnt_{}'.format(i)].rolling(3).mean()
                tt['rolling_mean_7_shift_{}'.format(i)] = tt['shift_arranged_cnt_{}'.format(i)].rolling(7).mean()
                tt['rolling_std_3_shift_{}'.format(i)] = tt['shift_arranged_cnt_{}'.format(i)].rolling(3).std()
                tt['rolling_std_7_shift_{}'.format(i)] = tt['shift_arranged_cnt_{}'.format(i)].rolling(7).std()
                tt.drop('shift_arranged_cnt_{}'.format(i), inplace=True, axis=1)
            window = tt['arranged_cnt'].rolling(window=14)
            mps_up = 3 * window.mean()
            tt['is_outliers'] = 0
            tt.loc[(tt['arranged_cnt'] > mps_up) & (tt['arranged_cnt'] > 10), 'is_outliers'] = 1
            tt.loc[tt.is_outliers == 1, 'arranged_cnt'] = 3 * tt['arranged_cnt'].rolling(window=14, min_periods=1,
                                                                                         center=True).mean()
            rst = tt[tt.index.isin(cur_label_date_dt_list)]
            rst_list.append(rst.reset_index().drop(['is_outliers'], axis=1))
        return pd.concat(rst_list)

    @staticmethod
    def his_sale_avg(raw_data):
        """
        历史销量平滑特征
        Args:
            raw_data: 原始数据

        Returns:
            处理好的特征数据
        """
        data = raw_data.sort_values('dt')
        sku_data_list = []

        for sku, sku_data in data.groupby(['sku_id', 'wh_id', 'bu_id']):
            df_sku = sku_data[sku_data.is_train == 1]
            arr_list = df_sku['arranged_cnt']

            # 计算7天/15天/30天移动平均值
            roll_7d_avg = arr_list.rolling(7, min_periods=1).mean()
            roll_14d_avg = arr_list.rolling(14, min_periods=1).mean()
            roll_21d_avg = arr_list.rolling(21, min_periods=1).mean()

            # 计算(训练集)加权均值
            his_avg = (roll_7d_avg + roll_14d_avg + roll_21d_avg) / 3.
            his_avg = np.concatenate([[his_avg.iloc[0]], his_avg])

            try:
                sku_data_list.append(pd.DataFrame({'sku_id': sku[0],
                                                   'wh_id': sku[1],
                                                   'bu_id': sku[2],
                                                   'dt': sku_data.dt,
                                                   'his_avg': his_avg}))
            except:
                log_print("his shape: {}, sku shape: {}， sku id: {}".format(his_avg.shape, sku_data.shape, sku[0]))
                return pd.DataFrame([])
        return pd.concat(sku_data_list)

    @staticmethod
    def train_lr_model(lr_train):
        """
        lr 模型训练
        Args:
            lr_train: 训练集

        Returns:
            训练好的模型
        """
        log_print("正在进行LR模型训练&预测...")
        X_train = lr_train[:, :-1]
        y_train = lr_train[:, -1]
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        return lr

    @staticmethod
    def lr_predict(model, df_test, ordered_cols):
        """
        lr 模型预测
        Args:
            model: 训练好的模型
            df_test: 测试集
            ordered_cols: 特征列表

        Returns:
            预测结果
        """
        # LR模型预测
        key_cols = ['wh_id', 'bu_id', 'sku_id', 'dt']
        X_test = df_test[ordered_cols].values[:, :-1]
        y_pred = model.predict(X_test)
        y_pred[y_pred < 0] = 0
        lr_forecast = df_test[key_cols]
        lr_forecast['prediction'] = y_pred
        lr_forecast['bu_id'] = lr_forecast['bu_id'].astype(int)
        lr_forecast['sku_id'] = lr_forecast['sku_id'].astype(int)
        lr_forecast['wh_id'] = lr_forecast['wh_id'].astype(int)
        log_print("LR模型训练&预测完成.")
        return lr_forecast

    @staticmethod
    def train_xgb_model(train_df, config):
        """
        训练 XGB 模型
        Args:
            train_df: 训练集
            config: 配置项字典

        Returns:
            训练好的模型
        """
        val_list = []
        ratio = 0.05
        for wh_id, wh_train in train_df.groupby('wh_id'):
            for cat2_name, cat_train in wh_train.groupby('cat2_name'):
                sample_num = len(cat_train) * ratio
                val_list.append(cat_train.sample(int(sample_num)))

        _val = pd.concat(val_list)
        _train = train_df.drop(index=_val.index, axis=0)

        X_train = _train[config['features']]
        y_train = _train['arranged_cnt']

        X_val = _val[config['features']]
        y_val = _val['arranged_cnt']

        begin_time = time.time()
        params = {'objective': 'reg:linear', 'learning_rate': 0.01, 'n_jobs': 8, 'nthread': 16, 'n_estimators': 600,
                  'max_depth': 4, 'min_child_weight': 0, 'subsample': 0.85, 'colsample_bytree': 0.85, 'seed': 42,
                  'disable_default_eval_metric': 0}
        bst = XGBRegressor(**params).fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            early_stopping_rounds=20,
            verbose=True
        )

        log_print('训练模型用时: {:.2f} min'.format((time.time() - begin_time) / 60.))
        return bst

    @staticmethod
    def xgb_predict(model, test_df, config):
        """
        XGB 模型预测
        Args:
            model: 训练好的模型
            test_df: 测试集
            config: 配置项字典

        Returns:
            XGB 模型预测结果
        """
        rst_df = pd.DataFrame()
        rst = model.predict(test_df[config['features']])
        rst_df[['sku_id', 'wh_id', 'bu_id', 'dt']] = test_df[['sku_id', 'wh_id', 'bu_id', 'dt']]
        rst_df['prediction'] = rst
        return rst_df

    @staticmethod
    def gen_lt_prediction(lt_data, config):
        """
        长尾预测
        Args:
            lt_data: 长尾数据集
            config: 配置项字典

        Returns:
            长尾预测结果
        """
        lt_data = pd.DataFrame(lt_data[1], columns=config['lt_data_cols']).replace('', np.nan)
        data = lt_data.sort_values('dt')
        fc_dt = config['current_date']
        result_list = []

        for key, values in data.groupby(['wh_id', 'bu_id', 'sku_id']):
            avg_val = values.avg_result.iloc[0]
            cat1_name = values.cat1_name.iloc[0]
            data = values.arranged_cnt

            if cat1_name == '蔬菜水果':
                if data.iloc[-1] <= 1:
                    pred_uc = data.iloc[-1]
                else:
                    pred_uc = data.iloc[-1] + np.random.normal(0, 0.1)

                if pred_uc < 0:
                    pred_uc = data.iloc[-1]
            else:
                pred_uc = avg_val

            _result = pd.DataFrame({'wh_id': [key[0]],
                                    'bu_id': [key[1]],
                                    'sku_id': [key[2]],
                                    'prediction': [pred_uc],
                                    'dt': [fc_dt]
                                    })
            result_list.append(_result)

        return pd.concat(result_list).to_dict('records')
