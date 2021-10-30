#!/bin/usr/env python3.6
# -*- coding: utf-8 -*-
# ---------------------------------
# ProjectName: data-forecast-spark
# Description: 3P销量预测特征制造
# Author: songzhen07
# CreateTime: 2021-04-14 14:38
# ---------------------------------

from datetime import timedelta
import pandas as pd
import numpy as np
from collections import defaultdict
import random
random.seed(42)
np.random.seed(42)

import warnings
warnings.filterwarnings('ignore')



class SalePopFeatureCreate:

    def __init__(self):
        pass


    def get_seasonal_data(self, df_train, dim_field, col_name):
        """
        分维度计算周期特征
        """
        # 各维度销量均值
        day_sale = df_train.groupby([dim_field, 'dt'], as_index=False).total_cnt.sum()

        # 星期销量均值/日常销量均值
        if col_name == 'day_abbr':
            new_col_name = dim_field[0:dim_field.rfind('_')] + '_week_avg'
            sale_mean = day_sale.groupby(dim_field, as_index=False).total_cnt.mean()
            day_week_sale = df_train.groupby([dim_field, col_name, 'dt'], as_index=False).total_cnt.sum()
            week_mean = day_week_sale.groupby([dim_field, col_name], as_index=False).total_cnt.mean()
            week_mean = week_mean.merge(sale_mean, how='left', on=[dim_field], suffixes=('_week', '_total'))
            week_mean[new_col_name] = week_mean['total_cnt_week'] / week_mean['total_cnt_total']
            return week_mean, new_col_name


        # 品牌销量均值/日常销量均值
        if col_name == 'brand_name':
            new_col_name = dim_field[0:dim_field.rfind('_')] + '_brand_avg'
            sale_mean = day_sale.groupby(dim_field, as_index=False).total_cnt.mean()
            day_brand_sale = df_train.groupby([dim_field, col_name, 'dt'], as_index=False).total_cnt.sum()
            brand_mean = day_brand_sale.groupby([dim_field, col_name], as_index=False).total_cnt.mean()
            brand_mean = brand_mean.merge(sale_mean, how='left', on=[dim_field], suffixes=('_brand', '_total'))
            brand_mean[new_col_name] = brand_mean['total_cnt_brand'] / brand_mean['total_cnt_total']
            return brand_mean, new_col_name


        # 节假日销量均值/日常销量均值
        if col_name == 'festival_name':
            new_col_name = dim_field[0:dim_field.rfind('_')] + '_fevl_avg'

            ## 取节假日前后各7天
            _delta = 7
            fetvl_dic = defaultdict(list)
            for f, d in df_train.groupby(col_name):
                if f == '无':
                    continue
                d = d.sort_values('dt')
                years = d.year.unique().tolist()
                for year in years:
                    yd = d[d.year == year]
                    fir_dt = yd.dt.iloc[0]
                    la_dt = yd.dt.iloc[-1]
                    be_fir_dt = fir_dt - timedelta(_delta)
                    af_la_dt = la_dt + timedelta(_delta)
                    fetvl_dic[f].append((be_fir_dt, fir_dt, la_dt, af_la_dt))

            ## 节假日期间维度销量均值
            day_fetvl_sale = df_train.groupby([dim_field, 'dt', col_name], as_index=False).total_cnt.sum()
            fetvl_mean = day_fetvl_sale.groupby([dim_field, col_name], as_index=False).total_cnt.mean()

            ## 节假日前后的销量均值
            fetvl_sale_list = []
            for dim, dim_data in df_train.groupby(dim_field):
                for f, r in fetvl_dic.items():
                    normal_sale = 0.
                    num = 0.
                    for ra in r:
                        df_normal_day = dim_data[((dim_data.dt >= ra[0]) & (dim_data.dt < ra[1])) |
                                                 ((dim_data.dt > ra[2]) & (dim_data.dt <= ra[3]))]
                        df_normal_day = df_normal_day[~df_normal_day[col_name].isin(fetvl_dic.keys())]
                        dim_day_sale = df_normal_day.groupby('dt', as_index=False).total_cnt.sum()
                        normal_sale += dim_day_sale.total_cnt.mean()
                        num += 1
                    fetvl_sale_list.append([dim, f, normal_sale / num])

            ## 评估各维度销量受节假日影响
            df_fetvl_sale = pd.DataFrame(fetvl_sale_list, columns=[dim_field, col_name, 'normal_sale_avg'])
            fetvl_mean = fetvl_mean.merge(df_fetvl_sale, how='left', on=[dim_field, col_name])
            fetvl_mean.loc[fetvl_mean[col_name] == '无', 'normal_sale_avg'] \
                = fetvl_mean.loc[fetvl_mean[col_name] == '无', 'total_cnt']
            fetvl_mean[new_col_name] = fetvl_mean['total_cnt'] / fetvl_mean['normal_sale_avg']
            fetvl_mean = fetvl_mean.replace('', 1.0)
            fetvl_mean = fetvl_mean.replace(np.inf, 1.0)
            return fetvl_mean, new_col_name


    def seasonal_count(self, total_data):
        """
        周期性统计特征
        """
        dump_cols = ['wh_id', 'bu_id', 'sku_id', 'dt']
        no_dump_data = total_data.drop_duplicates(subset=dump_cols, keep='first').sort_values('dt')
        df_train = no_dump_data[no_dump_data.is_train == 1]

        dim_field_list = ['wh_id', 'cat1_name', 'cat2_name']
        col_field_list = ['day_abbr', 'brand_name', 'festival_name']

        for dim_field in dim_field_list:
            for col_field in col_field_list:
                feature_data, feature_name = self.get_seasonal_data(df_train, dim_field, col_field)
                no_dump_data = no_dump_data.merge(feature_data[[dim_field, col_field, feature_name]],
                                                  how='left', on=[dim_field, col_field])

        # 缺失值处理
        null_cols = [col for col in no_dump_data.columns if (no_dump_data[col].isnull().sum() > 0 and 'cat2_' in col)]
        for col in null_cols:
            alt_col = 'cat1_' + col[5:]
            no_dump_data.loc[no_dump_data[col].isnull() == True, col] \
                = no_dump_data.loc[no_dump_data[col].isnull() == True, alt_col]

        cols = dump_cols + no_dump_data.columns.tolist()[-9:]
        total_data = total_data.merge(no_dump_data[cols], how='left', on=dump_cols)

        return total_data.fillna(1.)




    def his_sale_avg(self, total_data):
        """
        历史销量数据
        """
        dump_cols = ['wh_id', 'bu_id', 'sku_id', 'dt']
        data = total_data.drop_duplicates(subset=['wh_id', 'bu_id', 'sku_id', 'dt'], keep='first') \
            .sort_values('dt')

        sku_data_list = []

        for key, sku_data in data.groupby(['wh_id', 'bu_id', 'sku_id']):

            df_sku = sku_data[(sku_data.is_train == 1) & (sku_data.festival_name == '无')]

            ## 节假日期间上架商品不过滤
            if len(df_sku) == 0:
                df_sku = sku_data[sku_data.is_train == 1]

            train_and_test = pd.concat([df_sku, sku_data[sku_data.is_train == 0]])
            arr_list = df_sku['total_cnt']
            his_avg = arr_list.rolling(3, min_periods=1).mean()


            his_avg = np.concatenate([[his_avg.iloc[0]], his_avg])


            try:
                sku_data_list.append(pd.DataFrame({'wh_id': key[0],
                                                   'bu_id': key[1],
                                                   'sku_id': key[2],
                                                   'dt': train_and_test.dt,
                                                   'his_avg': his_avg}))
            except:
                print('{} 历史销量特征制造异常!'.format(sku))
                continue

        data = data.merge(pd.concat(sku_data_list), how='left', on=['wh_id', 'bu_id', 'sku_id', 'dt']) \
            .sort_values('dt')

        # 节假日销量处理
        _list = []
        _train = data[data.is_train == 1]
        _test = data[data.is_train == 0]
        for key, sku_data in _train.groupby(['wh_id', 'bu_id', 'sku_id']):
            df_na = sku_data[sku_data.his_avg.isnull() == True]
            try:
                f_dt = sku_data.dt.iloc[0]
                f_na_dt = df_na.dt.iloc[0]
                if f_dt == f_na_dt:
                    sku_data.loc[sku_data.dt == f_na_dt, 'his_avg'] \
                        = sku_data.loc[sku_data.dt == f_na_dt, 'total_cnt']
                    _list.append(sku_data)
                else:
                    _list.append(sku_data)
            except IndexError:
                _list.append(sku_data)
                continue

        new_train = pd.concat(_list)
        new_train['his_avg'] = new_train['his_avg'].fillna(method='ffill')
        result = pd.concat([new_train, _test]).sort_values('dt')

        new_train = pd.concat(_list)
        new_train['his_avg'] = new_train['his_avg'].fillna(method='ffill')
        result = pd.concat([new_train, _test]).sort_values('dt')
        # 考虑节假日影响之后的修正数据
        result.loc[result['festival_name'] != '无', 'his_avg'] \
            = result.loc[result['festival_name'] != '无']['his_avg'] \
              * result.loc[result['festival_name'] != '无']['cat1_fevl_avg']
        total_data = total_data.merge(result[dump_cols + ['his_avg']], how='left', on=dump_cols).dropna()

        return total_data




    def cnt_band_func(self, df_total_raw):
        """
        (根据销量范围)计算商品销量等级
        """
        # band级数和标签
        cut_num = 4
        label_names = np.arange(cut_num-1, -1, step=-1)
        df_train = df_total_raw[df_total_raw.is_train == 1]
        df_train = df_train.drop_duplicates(subset=['wh_id', 'bu_id', 'sku_id', 'dt'], keep='first') \
            .sort_values('dt')
        df_train['total_cnt'] = df_train['total_cnt'].astype(np.float32)

        cnt_band_data = []     # 二级品类band中间数据
        qpl_band_data = []     # 全品类band中间数据
        total_data = []        # 返回数据
        dt_set = set()         # 防止数据重复

        # n天窗口滑动
        dt_range = pd.date_range(df_train.dt.iloc[0], df_train.dt.iloc[-1])
        days = min(len(dt_range), 7)
        for i in range(days, len(dt_range)+1):
            # 窗口数据生成
            dt_window = dt_range[i-days:i]
            window_data = df_train[df_train.dt.isin(dt_window)]

            # 跳过无售卖记录期间
            if len(window_data) == 0:
                continue

            ## 计算窗口内二级品类下各sku销量均值
            cat2_group = window_data.groupby(['cat2_name', 'bu_id', 'sku_id'], as_index=False)['total_cnt'] \
                .agg({'arr_mean': np.mean})

            ## 计算窗口内各sku销量均值
            qpl_group = window_data.groupby(['bu_id', 'sku_id'], as_index=False)['total_cnt'] \
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
                                         'bu_id': group.bu_id,
                                         'sku_id': group.sku_id,
                                         'cnt_band_area': cnt_label.astype(np.int32)
                                         })
                cnt_band_data.append(df_label)

            ## 计算全品类下各sku销量band
            qpl_cnt_label = pd.cut(qpl_group.arr_mean.astype(np.float32), cut_num, precision=2, labels=label_names)
            qpl_df_label = pd.DataFrame({'dt': dt_key,
                                         'bu_id': qpl_group.bu_id,
                                         'sku_id': qpl_group.sku_id,
                                         'cnt_band_area_qpl': qpl_cnt_label.astype(np.int32)
                                         })
            qpl_band_data.append(qpl_df_label)

        cnt_band_data = pd.concat(cnt_band_data)
        qpl_band_data = pd.concat(qpl_band_data)

        # 合并特征列
        total_data_raw = df_total_raw.merge(cnt_band_data, how='left', on=['dt', 'cat2_name', 'bu_id', 'sku_id'])
        total_data_raw = total_data_raw.merge(qpl_band_data, how='left', on=['dt', 'bu_id', 'sku_id'])

        # 处理空值
        for sku, sku_data in total_data_raw.groupby(['bu_id', 'sku_id']):
            total_data.append(sku_data.fillna(method='bfill').fillna(method='ffill'))

        return pd.concat(total_data).fillna(3)




    def get_statistic_features(self, df_total_raw):
        """
        sku统计特征
        """
        df_total = df_total_raw.drop_duplicates(subset=['wh_id', 'bu_id', 'sku_id', 'dt'], keep='first') \
            .sort_values('dt')
        sku_data_list = []

        for key, sku_data in df_total.groupby(['bu_id', 'sku_id']):

            dt_array = sku_data.dt.unique()
            dt_lenth = len(dt_array)

            arr_list = sku_data[sku_data.is_train == 1]['total_cnt'].tolist()

            # 数据前移
            arr_array = np.concatenate([[arr_list[0]], arr_list])

            # 历史销量
            itvl_avg = arr_array.cumsum() / (pd.Series(arr_array).index + 1)

            # 历史销量指数均值
            itvl_ewm_avg = pd.Series(arr_array).ewm(span=3, min_periods=1).mean()

            # 滑动窗口统计值
            roll_arr_obj = pd.Series(arr_array).rolling(3, min_periods=1)

            itvl_roll_avg = roll_arr_obj.mean()     ## 销量滑动窗口均值
            itvl_roll_max = roll_arr_obj.max()      ## 销量滑动窗口最大值
            itvl_roll_min = roll_arr_obj.min()      ## 销量滑动窗口最小值

            sku_data_list.append(pd.DataFrame({'bu_id': [key[0]] * dt_lenth,
                                               'sku_id': [key[1]] * dt_lenth,
                                               'itvl_avg': itvl_avg,
                                               'itvl_ewm_avg': itvl_ewm_avg,
                                               'itvl_roll_avg': itvl_roll_avg,
                                               'itvl_roll_max': itvl_roll_max,
                                               'itvl_roll_min': itvl_roll_min,
                                               'dt': dt_array
                                               }))

        return df_total_raw.merge(pd.concat(sku_data_list), how='left', on=['bu_id', 'sku_id', 'dt'])



    def pro_statistic_features(self, total_data):
        """
        维度满减统计
        """
        dump_cols = ['wh_id', 'bu_id', 'sku_id', 'dt']
        data = total_data.drop_duplicates(subset=dump_cols, keep='first')
        dim_fields = ['cat1_name']
        act_fields = ['is_csu_redu', 'is_cir_redu']

        for dim in dim_fields:
            for act in act_fields:
                col_name = dim[0:dim.rfind('_')] + '_' + act[0:act.rfind('_')] + '_count'
                act_data = data.groupby(['dt', dim], as_index=False)[act].agg({col_name: np.sum})
                data = data.merge(act_data, how='left', on=['dt', dim])

        cols = dump_cols + data.columns.tolist()[-2:]
        return total_data.merge(data[cols], how='left', on=dump_cols).fillna(0.)