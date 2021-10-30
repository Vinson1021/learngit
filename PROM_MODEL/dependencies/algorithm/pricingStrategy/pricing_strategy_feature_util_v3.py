#!/bin/usr/env python3.6
# -*- coding: utf-8 -*-
# ---------------------------------
# ProjectName: data-forecast-spark
# Description: 量价模型衍生特征v3
# Author: songzhen07
# CreateTime: 2020-10-23 21:12
# ---------------------------------

import pandas as pd
import numpy as np
import itertools
import warnings
warnings.filterwarnings('ignore')

class FeatureUtil:

    def __init__(self):
        pass


    def week_sale_ratio(self, df_total_raw, df_train_total):
        """
        二级品类星期内销量占比
        """

        train_data = df_total_raw[df_total_raw.is_train == 1]

        ## 基准数据
        day_base = pd.DataFrame({'day_abbr': df_train_total['day_abbr'].unique()})
        cat2_base = pd.DataFrame({'cat2_name': df_train_total['cat2_name'].unique()})
        list_base = list(itertools.product(day_base.values.tolist(), cat2_base.values.tolist()))
        df_base = pd.DataFrame(list(map(lambda x : sum(x,[]), list_base)), columns=['day_abbr', 'cat2_name'])

        ## 各品类统计值
        dt_range = train_data.sort_values('dt').dt.unique()[-45:]
        part_train = train_data[train_data.dt.isin(dt_range)]
        count_data = part_train.groupby(['cat2_name', 'day_abbr'], as_index=False)['arranged_cnt'] \
            .agg({'day_avg': np.mean})
        count_data['day_avg'] += 0.01
        count_data['week_sum'] = count_data.groupby(['cat2_name']).transform(np.sum).iloc[:, -1]
        count_data['sale_ratio'] = count_data['day_avg'] / count_data['week_sum']

        ## 与基准数据合并，填补空值
        count_data = df_base.merge(count_data, how='left', on=['cat2_name', 'day_abbr'])
        count_data.loc[count_data['day_avg'].isnull() == True, ['day_avg', 'week_sum', 'sale_ratio']] = [1, 7, 1 / 7]

        ## 品类销量占比特征
        col = ['cat2_name', 'day_abbr', 'sale_ratio']
        df_total_raw = df_total_raw.merge(count_data[col], how='left', on=['cat2_name', 'day_abbr'])

        return df_total_raw


    def mov_arr_avg(self, raw_data):
        """
        移动平均销量特征
        """
        data = raw_data.sort_values('dt')
        sku_data_list = []

        for sku, sku_data in data.groupby('sku_id'):
            df_sku = sku_data[sku_data.is_train == 1]
            arr_array = df_sku['arranged_cnt']

            # 计算7天/15天/30天移动平均值
            roll_7d_avg = arr_array.rolling(7, min_periods=1).mean()
            roll_15d_avg = arr_array.rolling(15, min_periods=1).mean()
            roll_30d_avg = arr_array.rolling(30, min_periods=1).mean()

            # 计算(训练集)加权均值
            his_avg = (roll_7d_avg + roll_15d_avg + roll_30d_avg) / 3.

            # T+0数据
            his_avg_t0 = np.insert(np.array(his_avg), 0, his_avg.iloc[0], axis=0)

            # 构造T+1数据
            if len(his_avg_t0) < 7:
                avg_7d = np.mean(his_avg_t0)
            else:
                avg_7d = np.mean(his_avg_t0[-7:])

            if len(his_avg_t0) < 15:
                avg_15d = np.mean(his_avg_t0)
            else:
                avg_15d = np.mean(his_avg_t0[-15:])

            if len(his_avg_t0) < 30:
                avg_30d = np.mean(his_avg_t0)
            else:
                avg_30d = np.mean(his_avg_t0[-30:])

            # 最终数据
            mov_arr_avg = np.append(his_avg_t0, (avg_7d + avg_15d + avg_30d) / 3.)
            sku_data_list.append(pd.DataFrame({'sku_id': sku,
                                               'dt': sku_data.dt.unique(),
                                               'mov_arr_avg': mov_arr_avg
                                               }))
        # 最终结果
        data = data.merge(pd.concat(sku_data_list), how='left', on=['sku_id', 'dt'])
        return data


    def get_seasonal_data(self, df_train_total, dim_field, col_name):
        """
        季节性统计特征
        """
        # 各维度销量均值
        sale_mean = df_train_total.groupby(dim_field, as_index=False).arranged_cnt.mean()

        # 星期销量均值/日常销量均值
        if col_name == 'day_abbr':
            new_col_name = dim_field[0:dim_field.rfind('_')] + '_week_avg'
            week_mean = df_train_total.groupby([dim_field, col_name], as_index=False).arranged_cnt.mean()
            week_mean = week_mean.merge(sale_mean, how='left', on=[dim_field], suffixes=('_week', '_total'))
            week_mean[new_col_name] = week_mean['arranged_cnt_week'] / week_mean['arranged_cnt_total']
            return week_mean, new_col_name

        # 月份销量均值/日常销量均值
        if col_name == 'month':
            new_col_name = dim_field[0:dim_field.rfind('_')] + '_month_avg'
            month_mean = df_train_total.groupby([dim_field, col_name], as_index=False).arranged_cnt.mean()
            month_mean = month_mean.merge(sale_mean, how='left', on=[dim_field], suffixes=('_week', '_total'))
            month_mean[new_col_name] = month_mean['arranged_cnt_week'] / month_mean['arranged_cnt_total']
            return month_mean, new_col_name

        # 品牌销量均值/日常销量均值
        if col_name == 'brand_name':
            new_col_name = dim_field[0:dim_field.rfind('_')] + '_brand_avg'
            brand_mean = df_train_total.groupby([dim_field, col_name], as_index=False).arranged_cnt.mean()
            brand_mean = brand_mean.merge(sale_mean, how='left', on=[dim_field], suffixes=('_week', '_total'))
            brand_mean[new_col_name] = brand_mean['arranged_cnt_week'] / brand_mean['arranged_cnt_total']
            return brand_mean, new_col_name

        # 节假日销量均值/日常销量均值
        if col_name == 'festival_name':
            new_col_name = dim_field[0:dim_field.rfind('_')] + '_fevl_avg'
            fevl_mean = df_train_total.groupby([dim_field, col_name], as_index=False).arranged_cnt.mean()
            fevl_mean.rename(columns={'arranged_cnt': new_col_name}, inplace=True)
            fevl_mean = fevl_mean.merge(sale_mean)
            fevl_mean[new_col_name] = fevl_mean[new_col_name] / fevl_mean['arranged_cnt']
            del fevl_mean['arranged_cnt']
            return fevl_mean, new_col_name

        # 西方节假日销量均值/日常销量均值
        if col_name == 'western_festival_name':
            new_col_name = dim_field[0:dim_field.rfind('_')] + '_wfevl_avg'
            wfevl_mean = df_train_total.groupby([dim_field, col_name], as_index=False).arranged_cnt.mean()
            wfevl_mean.rename(columns={'arranged_cnt': new_col_name}, inplace=True)
            wfevl_mean = wfevl_mean.merge(sale_mean)
            wfevl_mean[new_col_name] = wfevl_mean[new_col_name] / wfevl_mean['arranged_cnt']
            del wfevl_mean['arranged_cnt']
            return wfevl_mean, new_col_name


    def seasonal_count(self, df_total_raw, df_train_total):
        """
        周期性统计特征
        """
        dim_field_list = ['cat1_name', 'cat2_name']
        col_field_list = ['day_abbr', 'month', 'brand_name', 'festival_name', 'western_festival_name']

        for dim_field in dim_field_list:
            for col_field in col_field_list:
                feature_data, feature_name = self.get_seasonal_data(df_train_total, dim_field, col_field)
                df_total_raw = df_total_raw.merge(feature_data[[dim_field, col_field, feature_name]],
                                                  how='left', on=[dim_field, col_field])
        return df_total_raw


    def byrs_count(self, df_total_raw, df_train_total, col_list=None):
        """
        商户数量统计特征
        """
        for col in col_list:
            # 要转化的商户特征
            week_col_name = col + '_week_avg'
            month_col_name = col + '_month_avg'
            fevl_col_name = col + '_fevl_avg'
            wfevl_col_name = col + '_wfevl_avg'

            # 商户数均值
            _mean = df_train_total[col].mean()

            # 星期商户均值/日常商户均值
            _week_mean = df_train_total.groupby(['day_abbr'], as_index=False)[[col]].mean()
            _week_mean[week_col_name] = _week_mean.iloc[:, -1] / _mean

            # 月份商户均值/日常商户均值
            _month_mean = df_train_total.groupby(['month'], as_index=False)[[col]].mean()
            _month_mean[month_col_name] = _month_mean.iloc[:, -1] / _mean

            # 节假日商户均值/日常商户均值
            _fevl_mean = df_train_total.groupby(['festival_name'], as_index=False)[[col]].mean()
            _fevl_mean[fevl_col_name] = _fevl_mean.iloc[:, -1] / _mean

            # 品类西方节假日商户均值/日常商户均值
            _wfevl_mean = df_train_total.groupby(['western_festival_name'], as_index=False)[[col]].mean()
            _wfevl_mean[wfevl_col_name] = _wfevl_mean.iloc[:, -1] / _mean

            # 合并数据
            df_total_raw = df_total_raw.merge(_week_mean[['day_abbr', week_col_name]],
                                              how='left', on=['day_abbr'])
            df_total_raw = df_total_raw.merge(_month_mean[['month', month_col_name]],
                                              how='left', on=['month'])
            df_total_raw = df_total_raw.merge(_fevl_mean[['festival_name', fevl_col_name]],
                                              how='left', on=['festival_name'])
            df_total_raw = df_total_raw.merge(_wfevl_mean[['western_festival_name', wfevl_col_name]],
                                              how='left', on=['western_festival_name'])
        # 返回去除原始商户特征的数据
        return df_total_raw.fillna(0.).drop(col_list, axis=1)


    def cnt_band_func(self, df_total_raw):
        """
        根据销量范围计算sku的
        销量band
        """
        # band级数和标签
        cut_num = 4
        label_names = np.arange(cut_num-1, -1, step=-1)
        df_train = df_total_raw[df_total_raw.is_train == 1].sort_values('dt')

        cnt_band_data = []     # 二级品类band中间数据
        qpl_band_data = []     # 全品类band中间数据
        total_data = []        # 返回数据
        dt_set = set()         # 防止数据重复

        # 以7天为窗口滑动
        dt_range = pd.date_range(df_train.dt.iloc[0], df_train.dt.iloc[-1])
        for i in range(7, len(dt_range)+1):
            # 窗口数据生成
            dt_window = dt_range[i-7:i]
            window_data = df_train[df_train.dt.isin(dt_window)]

            # 跳过无售卖记录期间
            if len(window_data) == 0:
                continue

            ## 计算窗口内二级品类下各sku销量均值
            cat2_group = window_data.groupby(['cat2_name','sku_id'], as_index=False)['arranged_cnt'] \
                .agg({'arr_mean': np.mean})

            ## 计算窗口内各sku销量均值
            qpl_group = window_data.groupby('sku_id', as_index=False)['arranged_cnt'] \
                .agg({'arr_mean': np.mean})

            ## 用于关联的dt键
            dt_key = window_data.dt.iloc[-1]

            ## 保存已经滑过的窗口日期
            if dt_key in dt_set:
                continue
            dt_set.add(dt_key)

            cat2_group['dt'] = dt_key
            qpl_group['dt'] = dt_key

            ## 计算二级品类下各sku销量band
            for cat2, group in cat2_group.groupby('cat2_name'):
                cnt_label = pd.cut(group.arr_mean.astype(np.float32), cut_num, precision=2, labels=label_names)
                df_label = pd.DataFrame({'dt': dt_key,
                                         'cat2_name': cat2,
                                         'sku_id': group.sku_id,
                                         'cnt_band_area': cnt_label.astype(np.int32)
                                         })
                cnt_band_data.append(df_label)

            ## 计算全品类下各sku销量band
            qpl_cnt_label = pd.cut(qpl_group.arr_mean.astype(np.float32), cut_num, precision=2, labels=label_names)
            qpl_df_label = pd.DataFrame({'dt': dt_key,
                                         'sku_id': qpl_group.sku_id,
                                         'cnt_band_area_qpl': qpl_cnt_label.astype(np.int32)
                                         })
            qpl_band_data.append(qpl_df_label)

        cnt_band_data = pd.concat(cnt_band_data)
        qpl_band_data = pd.concat(qpl_band_data)

        # 合并特征列
        total_data_raw = df_total_raw.merge(cnt_band_data, how='left', on=['dt', 'cat2_name', 'sku_id'])
        total_data_raw = total_data_raw.merge(qpl_band_data, how='left', on=['dt', 'sku_id'])

        # 处理空值
        for sku, sku_data in total_data_raw.groupby('sku_id'):
            total_data.append(sku_data.fillna(method='bfill').fillna(method='ffill'))

        return pd.concat(total_data)