#!/bin/usr/env python3.6
# -*- coding: utf-8 -*-
# ---------------------------------
# ProjectName: data-forecast-server
# Description: 
# Author: songzhen07
# CreateTime: 2021-03-23 11:29
# ---------------------------------

import time
from datetime import timedelta
import pandas as pd
import numpy as np
import warnings
import os
import multiprocessing
from sklearn.linear_model import LinearRegression
import random
import traceback

random.seed(42)
np.random.seed(42)
from dependencies.common.forecast_utils import log_print

warnings.filterwarnings('ignore')


class SalePopBaselineForecast:

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

    def holiday_label_process(self, data, dt_field):
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
            mid_results.append(pool.apply_async(self.choose_data, args=(part, part_train)))
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

    @staticmethod
    def get_avg_result(all_data, config):
        """
        历史均值计算
        :param all_data: 训练数据
        :return: 历史n日销量均值
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

    def lt_data_prepare(self, lr_forecast):
        """
        长尾数据准备
        :return: 长尾数据
        """
        # LR未预测SKU
        tr_skus = self.train_data \
            .drop_duplicates(subset=['wh_id', 'bu_id', 'sku_id'], keep='first')[['wh_id', 'bu_id', 'sku_id']]
        lr_skus = lr_forecast[['wh_id', 'bu_id', 'sku_id']] \
            .drop_duplicates(subset=['wh_id', 'bu_id', 'sku_id'], keep='first')
        lr_skus['lr'] = 1

        merged = tr_skus.merge(lr_skus, how='left', on=['wh_id', 'bu_id', 'sku_id'])
        leave_skus = merged[merged.lr.isnull() == True].drop('lr', axis=1)

        # 和长尾合并
        lt_skus = leave_skus \
            .merge(self.longtail_data[['wh_id', 'bu_id', 'sku_id']], how='outer', on=['wh_id', 'bu_id', 'sku_id'])

        cols = ['dt', 'wh_id', 'bu_id', 'sku_id', 'cat1_name', 'arranged_cnt']
        cnt_longtail = self.raw_data[self.raw_data.is_train == 1] \
            .merge(lt_skus, how='inner', on=lt_skus.columns.tolist())[cols] \
            .sort_values('dt')

        avg_data = self.get_avg_result(cnt_longtail)
        lt_data = cnt_longtail.merge(avg_data, how='inner', on=['bu_id', 'wh_id', 'sku_id', 'cat1_name'])
        return lt_data

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

    def train_and_forecast(self, df):
        """
        训练 & 预测
        :return: None
        """
        df['dt'] = pd.to_datetime(df['dt'], format='%Y%m%d')
        ## LR模型训练
        self.lr_data_prepare(df)
        log_print("正在进行LR模型训练&预测...")
        X_train = self.lr_train[:, :-1]
        y_train = self.lr_train[:, -1]
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        ## LR模型预测
        key_cols = ['wh_id', 'bu_id', 'sku_id', 'dt']
        X_test = self.lr_test[:, :-1]
        y_pred = lr.predict(X_test)
        y_pred[y_pred < 0] = 0
        lr_forecast = self.df_test[key_cols]
        lr_forecast['prediction'] = y_pred
        log_print("LR模型训练&预测完成.")

        log_print("正在进行长尾预测...")
        lt_data = self.lt_data_prepare(lr_forecast)
        lt_forecast = self.longtail_pre(lt_data)
        log_print("长尾预测完成.")

        final_rst = pd.concat([lr_forecast, lt_forecast]) \
            .drop_duplicates(subset=['wh_id', 'bu_id', 'sku_id'], keep='first')
        final_rst['date'] = final_rst['dt'].apply(lambda x: x.strftime('%Y%m%d'))

        return final_rst

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
        df_test_output = df_test[list(set(config['key_cols']+config['ordered_cols']))]
        train_data_output = train_data.drop_duplicates(subset=['wh_id', 'bu_id', 'sku_id'], keep='first')[config['train_data_cols']]
        longtail_data_output = longtail_data[config['longtail_data_cols']]
        raw_data['dt'] = raw_data['dt'].dt.strftime("%Y%m%d")
        raw_data_output = raw_data[config['raw_data_cols']]
        return ('df_train', df_train_output.to_dict('records')), \
               ('train_data', train_data_output.to_dict('records')), \
               ('longtail_data', longtail_data_output.to_dict('records')), \
               ('raw_data', raw_data_output.to_dict('records')), \
               ('df_test', df_test_output.to_dict('records'))

    @staticmethod
    def train_lr_model(lr_train):
        """
        lr model training
        Args:
            lr_train: training dataset

        Returns:
            lr: lr model
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
        lr model prediction
        Args:
            model: trained lr model
            df_test: raw testing dataset
            ordered_cols: feature cols

        Returns:
            lr_forecast: prediction dataframe
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
    def preprocess_lt_data(train_data, longtail_data, raw_data, lr_forecast):
        """
        longtail data preprocessing
        Args:
            train_data: training dataset
            longtail_data: longtail dataset
            raw_data: raw dataset
            lr_forecast: LR model prediction

        Returns:
            lt_data: preprocessed longtail data
        """
        # LR未预测SKU
        tr_skus = train_data \
            .drop_duplicates(subset=['wh_id', 'bu_id', 'sku_id'], keep='first')[['wh_id', 'bu_id', 'sku_id']]
        lr_skus = lr_forecast[['wh_id', 'bu_id', 'sku_id']] \
            .drop_duplicates(subset=['wh_id', 'bu_id', 'sku_id'], keep='first')
        lr_skus['lr'] = 1

        merged = tr_skus.merge(lr_skus, how='left', on=['wh_id', 'bu_id', 'sku_id'])
        leave_skus = merged[merged.lr.isnull() == True].drop('lr', axis=1)

        # 和长尾合并
        lt_skus = leave_skus \
            .merge(longtail_data[['wh_id', 'bu_id', 'sku_id']], how='outer', on=['wh_id', 'bu_id', 'sku_id'])

        cols = ['dt', 'wh_id', 'bu_id', 'sku_id', 'cat1_name', 'arranged_cnt']
        cnt_longtail = raw_data[raw_data.is_train == 1] \
            .merge(lt_skus, how='inner', on=lt_skus.columns.tolist())[cols] \
            .sort_values('dt')

        avg_data = SalePopBaselineForecast.get_avg_result(cnt_longtail)
        lt_data = cnt_longtail.merge(avg_data, how='inner', on=['bu_id', 'wh_id', 'sku_id', 'cat1_name'])
        return lt_data

    @staticmethod
    def gen_lt_prediction(lt_data, config):
        """
        longtail prediction generation
        Args:
            lt_data: longtail dataset
            config: config dict

        Returns:
            longtail predictions
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
