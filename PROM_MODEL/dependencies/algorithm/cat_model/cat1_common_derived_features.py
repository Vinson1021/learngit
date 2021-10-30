#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------
# File  : cat1_common_derived_features.py
# Author: liushichang
# Date  : 2021/3/23
# Desc  : 衍生特征
# Contact : liushichang@meituan.com
# ----------------------------------
import itertools

import numpy as np
import pandas as pd


class DerivedFeature(object):
    def __init__(self, cat1_id, dt_field='dt'):
        self.cat1_id = cat1_id
        self.dt_field = dt_field

    @staticmethod
    def get_day_possion_pred(total_data, total_train_data, longtail_data):
        """
        获取泊松分布预测值
        :param total_data: 原始数据
        :param total_train_data: 训练数据
        :param longtail_data: 长尾数据
        :return: 泊松分布结果
        """

        results = []
        longtail_results = []

        for wh_id, df_total_raw in total_data.groupby('wh_id'):

            # step1: 非长尾数据处理
            train_data = df_total_raw[df_total_raw.is_train == 1]

            # 基准数据
            day_base = pd.DataFrame({'day_abbr': total_train_data['day_abbr'].unique()})
            cat2_base = pd.DataFrame({'cat2_name': total_train_data['cat2_name'].unique()})
            list_base = list(itertools.product(day_base.values.tolist(), cat2_base.values.tolist()))
            df_base = pd.DataFrame(list(map(lambda x: sum(x, []), list_base)), columns=['day_abbr', 'cat2_name'])

            # 各品类统计值
            dt_range = train_data.sort_values('dt').dt.unique()[-45:]
            part_train = train_data[train_data.dt.isin(dt_range)]
            count_data = part_train.groupby(['cat2_name', 'day_abbr'], as_index=False)['arranged_cnt'] \
                .agg({'day_avg': np.mean})
            count_data['day_avg'] += 0.01
            count_data['week_sum'] = count_data.groupby(['cat2_name']).transform(np.sum).iloc[:, -1]
            count_data['sale_ratio'] = count_data['day_avg'] / count_data['week_sum']

            # 与基准数据合并，填补空值
            count_data = df_base.merge(count_data, how='left', on=['cat2_name', 'day_abbr'])
            count_data.loc[count_data['day_avg'].isnull() == True, ['day_avg', 'week_sum', 'sale_ratio']] \
                = [1, 7, 1 / 7]

            # 品类销量占比特征
            col = ['cat2_name', 'day_abbr', 'sale_ratio']
            df_total_raw = df_total_raw.merge(count_data[col], how='left', on=['cat2_name', 'day_abbr'])

            # 汇总非长尾数据
            results.append(df_total_raw)

            # step2: 泊松分布处理
            possion_data = longtail_data[longtail_data.wh_id == wh_id]
            if len(possion_data) == 0:
                continue

            # 按天拆分泊松分布预测值
            merged_results = count_data.merge(possion_data, how='left', on='cat2_name')
            merged_results['possion_val_day'] = merged_results['possion_val'] * merged_results['sale_ratio']
            possion_data = merged_results[['bu_id', 'wh_id', 'sku_id', 'day_abbr', 'possion_val_day']].dropna()

            # 与测试日期合并
            test_date = df_total_raw[df_total_raw.is_train == 0][['date', 'day_abbr']]
            test_date.drop_duplicates(subset=['date'], keep='first', inplace=True)
            possion_data = possion_data.merge(test_date, how='left', on=['day_abbr'])[
                ['bu_id', 'wh_id', 'sku_id', 'date', 'possion_val_day']
            ]

            # 汇总长尾数据
            longtail_results.append(possion_data)

        return pd.concat(results), pd.concat(longtail_results)

    @staticmethod
    def get_seasonal_data(df_train_total, dim_field, col_name):
        """
        分维度计算周期特征
        :param df_train_total: 训练数据
        :param dim_field: 维度字段
        :param col_name: 周期字段
        :return: 特征数据
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

    @staticmethod
    def seasonal_count(df_total_raw, df_train_total):
        """
        周期性统计特征
        :param df_total_raw: 原始数据
        :param df_train_total: 训练数据
        :return: 周期性特征数据
        """
        dim_field_list = ['wh_id', 'cat1_name', 'cat2_name']
        col_field_list = ['day_abbr', 'month', 'brand_name', 'festival_name', 'western_festival_name']

        for dim_field in dim_field_list:
            for col_field in col_field_list:
                feature_data, feature_name = DerivedFeature.get_seasonal_data(df_train_total, dim_field, col_field)
                df_total_raw = df_total_raw.merge(feature_data[[dim_field, col_field, feature_name]],
                                                  how='left', on=[dim_field, col_field])
        return df_total_raw.fillna(0.)

    @staticmethod
    def get_sale_ratio(total_data, total_train_data):
        """
        获取销售比率
        :param total_data: 原始数据
        :param total_train_data: 训练数据
        :return: 销售比率结果
        """

        results = []
        for wh_id, df_total_raw in total_data.groupby('wh_id'):
            # step1: 非长尾数据处理
            train_data = df_total_raw[df_total_raw.is_train == 1]

            # 基准数据
            day_base = pd.DataFrame({'day_abbr': total_train_data['day_abbr'].unique()})
            cat2_base = pd.DataFrame({'cat2_name': total_train_data['cat2_name'].unique()})
            list_base = list(itertools.product(day_base.values.tolist(), cat2_base.values.tolist()))
            df_base = pd.DataFrame(list(map(lambda x: sum(x, []), list_base)), columns=['day_abbr', 'cat2_name'])

            # 各品类统计值
            dt_range = train_data.sort_values('dt').dt.unique()[-45:]
            part_train = train_data[train_data.dt.isin(dt_range)]
            count_data = part_train.groupby(['cat2_name', 'day_abbr'], as_index=False)['arranged_cnt'] \
                .agg({'day_avg': np.mean})
            count_data['day_avg'] += 0.01
            count_data['week_sum'] = count_data.groupby(['cat2_name']).transform(np.sum).iloc[:, -1]
            count_data['sale_ratio'] = count_data['day_avg'] / count_data['week_sum']

            # 与基准数据合并，填补空值
            count_data = df_base.merge(count_data, how='left', on=['cat2_name', 'day_abbr'])
            count_data.loc[count_data['day_avg'].isnull() == True, ['day_avg', 'week_sum', 'sale_ratio']] \
                = [1, 7, 1 / 7]

            # 品类销量占比特征
            col = ['cat2_name', 'day_abbr', 'sale_ratio']
            df_total_raw = df_total_raw.merge(count_data[col], how='left', on=['cat2_name', 'day_abbr'])

            results.append(df_total_raw)

        return pd.concat(results)

    @staticmethod
    def byrs_count(df_total_raw, df_train_total, col_list=None):
        """
        商户数量统计特征
        :param df_total_raw: 原始数据
        :param df_train_total: 原始训练数据
        :param col_list: 商户原始特征(返回时废弃)
        :return: 商户数量统计特征
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

    @staticmethod
    def cnt_band_func(df_total_raw):
        """
        (根据销量范围)计算商品销量等级
        :param df_total_raw: 原始数据
        :return: 销量等级
        """
        # band级数和标签
        cut_num = 4
        label_names = np.arange(cut_num - 1, -1, step=-1)
        df_train = df_total_raw[df_total_raw.is_train == 1].sort_values('dt')

        cnt_band_data = []  # 二级品类band中间数据
        qpl_band_data = []  # 全品类band中间数据
        total_data = []  # 返回数据
        dt_set = set()  # 防止数据重复

        # 以7天为窗口滑动
        dt_range = pd.date_range(df_train.dt.iloc[0], df_train.dt.iloc[-1])
        for i in range(7, len(dt_range) + 1):
            # 窗口数据生成
            dt_window = dt_range[i - 7:i]
            window_data = df_train[df_train.dt.isin(dt_window)]

            # 跳过无售卖记录期间
            if len(window_data) == 0:
                continue

            # 计算窗口内二级品类下各sku销量均值
            cat2_group = window_data.groupby(['cat2_name', 'sku_id'], as_index=False)['arranged_cnt'] \
                .agg({'arr_mean': np.mean})

            # 计算窗口内各sku销量均值
            qpl_group = window_data.groupby('sku_id', as_index=False)['arranged_cnt'] \
                .agg({'arr_mean': np.mean})

            # 用于关联的dt键
            dt_key = window_data.dt.iloc[-1]

            # 保存已经滑过的窗口日期
            if dt_key in dt_set:
                continue
            dt_set.add(dt_key)

            cat2_group['dt'] = dt_key
            qpl_group['dt'] = dt_key

            # 计算二级品类下各sku销量band
            for cat2, group in cat2_group.groupby('cat2_name'):
                cnt_label = pd.cut(group.arr_mean.astype(np.float32), cut_num, precision=2, labels=label_names)
                df_label = pd.DataFrame({'dt': dt_key,
                                         'cat2_name': cat2,
                                         'sku_id': group.sku_id,
                                         'cnt_band_area': cnt_label.astype(np.int32)
                                         })
                cnt_band_data.append(df_label)

            # 计算全品类下各sku销量band
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

    @staticmethod
    def get_statistic_features(df_total_raw):
        """
        sku统计特征
        :param df_total_raw: 原始数据
        :return: sku统计特征
        """
        df_total_raw = df_total_raw.sort_values('dt')
        sku_data_list = []

        for sku, sku_data in df_total_raw.groupby('sku_id'):
            dt_array = sku_data.dt.unique()
            dt_lenth = len(dt_array)

            arr_list = sku_data[sku_data.is_train == 1]['arranged_cnt'].tolist()

            # 将数据前移20天
            arr_array = np.concatenate([[arr_list[0]] * 20, arr_list])

            # 历史销量
            itvl_avg = arr_array.cumsum() / (pd.Series(arr_array).index + 1)

            # 历史销量指数均值
            itvl_ewm_avg = pd.Series(arr_array).ewm(span=4, min_periods=1).mean()

            # 滑动窗口统计值
            roll_arr_obj = pd.Series(arr_array).rolling(7, min_periods=1)

            itvl_roll_avg = roll_arr_obj.mean()  # 销量滑动窗口均值
            itvl_roll_max = roll_arr_obj.max()  # 销量滑动窗口最大值
            itvl_roll_min = roll_arr_obj.min()  # 销量滑动窗口最小值

            sku_data_list.append(pd.DataFrame({'sku_id': [sku] * dt_lenth,
                                               'itvl_avg': itvl_avg,
                                               'itvl_ewm_avg': itvl_ewm_avg,
                                               'itvl_roll_avg': itvl_roll_avg,
                                               'itvl_roll_max': itvl_roll_max,
                                               'itvl_roll_min': itvl_roll_min,
                                               'dt': dt_array
                                               }))

        return df_total_raw.merge(pd.concat(sku_data_list), how='left', on=['sku_id', 'dt'])

    @staticmethod
    def pro_statistic_features(data):
        """
        促销统计特征
        :param data: 原始数据
        :return: 促销统计特征
        """
        dim_fields = ['brand_name', 'cat1_name', 'cat2_name']
        act_fields = ['seq_num', 'pro_num', 'csu_redu_num']

        for dim in dim_fields:
            for act in act_fields:
                col_name = dim[0:dim.rfind('_')] + '_' + act[0:act.rfind('_')] + '_count'
                act_data = data.groupby(['dt', dim], as_index=False)[act].agg({col_name: np.sum})
                data = data.merge(act_data, how='left', on=['dt', dim])
        return data.fillna(0.)

    @staticmethod
    def price_derived_features(df_total_raw):
        """
        价格衍生特征制造
        :param df_total_raw: 原始数据
        :return: 价格特征
        """
        # 差价特征
        # 促销活动差价
        df_total_raw['sp_pdiff'] = df_total_raw['seq_price'] - df_total_raw['discount_price']
        # 真实差价
        df_total_raw['real_pdiff'] = df_total_raw['csu_origin_price'] - df_total_raw['w_price']

        # 折扣力度
        price_diff_seq = df_total_raw['csu_origin_price'] - df_total_raw['seq_price']
        price_diff_pro = df_total_raw['csu_origin_price'] - df_total_raw['discount_price']
        price_diff_real = df_total_raw['csu_origin_price'] - df_total_raw['w_price']

        # 秒杀折扣
        df_total_raw['seq_rebate'] = price_diff_seq / df_total_raw['csu_origin_price']
        # 促销折扣
        df_total_raw['pro_rebate'] = price_diff_pro / df_total_raw['csu_origin_price']
        # 实际折扣
        df_total_raw['real_rebate'] = price_diff_real / df_total_raw['csu_origin_price']

        # 竞品折扣
        # 竞品秒杀力度
        seq_data = df_total_raw.groupby(['dt', 'cat2_name'], as_index=False)['seq_rebate'] \
            .agg({'max_seq_rebate': np.max,
                  'mean_seq_rebate': np.mean})
        # 竞品单品促销力度
        pro_data = df_total_raw.groupby(['dt', 'cat2_name'], as_index=False)['pro_rebate'] \
            .agg({'max_pro_rebate': np.max,
                  'mean_pro_rebate': np.mean})
        # 竞品实际折扣
        real_data = df_total_raw.groupby(['dt', 'cat2_name'], as_index=False)['real_rebate'] \
            .agg({'max_real_rebate': np.max,
                  'mean_real_rebate': np.mean})

        # 合并数据
        rebate_data = seq_data.merge(pro_data, how='outer', on=['dt', 'cat2_name'])
        rebate_data = rebate_data.merge(real_data, how='outer', on=['dt', 'cat2_name'])
        df_total_raw = df_total_raw.merge(rebate_data, how='left', on=['dt', 'cat2_name']) \
            .fillna(0.) \
            .sort_values('dt')

        # 滑动（加权）价格特征
        sku_data_list = []
        for sku, sku_data in df_total_raw.groupby('sku_id'):
            df_sku = sku_data[sku_data.is_train == 1]

            # 7天滑动(加权)价格均值
            price_7d_roll = df_sku['w_price'].rolling(7, min_periods=1).mean()

            # T+0数据
            mov_price_avg = np.insert(np.array(price_7d_roll), 0, price_7d_roll.iloc[0], axis=0)

            # 外推覆盖未来20天
            price_list = np.append(df_sku['w_price'], price_7d_roll.iloc[-1])
            for i in range(19):
                if len(price_list) < 7:
                    avg_7d = np.mean(price_list)
                else:
                    avg_7d = np.mean(price_list[-7:])
                price_list = np.append(price_list, avg_7d)
            mov_price_avg = np.concatenate([mov_price_avg, price_list[-19:]])

            sku_data_list.append(pd.DataFrame({'sku_id': sku,
                                               'dt': sku_data.dt.unique(),
                                               'mov_price_avg': mov_price_avg
                                               }))
        # 合并sku数据
        df_total_raw = df_total_raw.merge(pd.concat(sku_data_list), how='left', on=['sku_id', 'dt'])
        # 原价/滑动价格差价
        df_total_raw['mov_delta_cprice'] = df_total_raw['csu_origin_price'] - df_total_raw['mov_price_avg']
        # 加权价/滑动价格差价
        df_total_raw['mov_delta_wprice'] = df_total_raw['w_price'] - df_total_raw['mov_price_avg']

        return df_total_raw

    @staticmethod
    def get_price_elasticity(df_total_raw):
        """
        价格弹性计算
        :param df_total_raw: 原始数据
        :return: 价格弹性系数
        """
        df_total_raw = df_total_raw.sort_values('dt')

        # 赋初始值
        ed_dic = {sku: 0. for sku in df_total_raw.sku_id.unique()}

        # 过滤满减数据
        df_ed = df_total_raw[(df_total_raw.is_train == 1) &
                             (df_total_raw.csu_redu_num == 0) &
                             (df_total_raw.arranged_cnt > 0)]

        # 计算价格弹性系数
        for sku, sku_train in df_ed.groupby('sku_id'):
            # 数据量限制
            if len(sku_train) < 15:
                continue
            # 使用滑动均值价格
            price_and_arr = sku_train[['w_price', 'arranged_cnt']].rolling(15, min_periods=15).mean()

            # 价格和销量变动值
            _delta = (price_and_arr - price_and_arr.shift(1)).rename(columns={'w_price': 'delta_price',
                                                                              'arranged_cnt': 'delta_arr'})
            # 合并两部分数据
            _delta = price_and_arr.join(_delta.shift(-1)).dropna()
            _delta = _delta[_delta.delta_price != 0]

            # 过滤价格无波动的sku
            if len(_delta) == 0:
                continue

            # 计算价格弹性系数
            darr = _delta['delta_arr']
            arr = _delta['arranged_cnt']
            dp = _delta['delta_price']
            p = _delta['w_price']
            ed = -(darr / arr) / (dp / p)

            # 过滤弹性系数中的异常值
            Q1 = np.percentile(ed, 25)
            Q3 = np.percentile(ed, 75)
            step = (Q3 - Q1) * 1.5
            mean_ed = ed[(ed <= Q3 + step) & (ed >= Q1 - step)].mean()

            # 更新弹性系数
            if not np.isnan(mean_ed):
                ed_dic[sku] = mean_ed

        # 生成价格弹性特征
        ed_list = [[sku_id, ed_dic[sku_id]] for sku_id in ed_dic.keys()]
        df_ed = pd.DataFrame(ed_list, columns=['sku_id', 'ed_val'])

        return df_total_raw.merge(df_ed, how='left', on=['sku_id']).sort_values('dt')

    @staticmethod
    def ed_pred_sale(df_total_raw):
        """
        价格弹性销量
        :param df_total_raw: (含价格弹性特征的)原始数据
        :return: 价格弹性销量
        """
        df_total_raw = df_total_raw.sort_values('dt')
        sku_data_list = []

        for sku, sku_data in df_total_raw.groupby('sku_id'):
            # 获取弹性系数
            ed_val = sku_data['ed_val'].unique()[0]

            _shift = sku_data[['w_price', 'arranged_cnt']].shift(1) \
                .rename(columns={'w_price': 'last_price',
                                 'arranged_cnt': 'last_arr'})
            _shift = _shift.join(sku_data[['w_price', 'arranged_cnt', 'is_train']])

            # 修正测试集数据
            last_price = sku_data[sku_data.is_train == 1]['w_price'].iloc[-1]
            last_arr = sku_data[sku_data.is_train == 1]['arranged_cnt'].iloc[-1]
            _shift.loc[_shift.is_train == 0, 'last_price'] = last_price
            _shift.loc[_shift.is_train == 0, 'last_arr'] = last_arr

            # 价格一阶差分
            _shift['delta_price'] = _shift['w_price'] - _shift['last_price']

            # 填补缺失值
            _shift.loc[_shift['last_price'].isnull() == True, 'last_price'] \
                = _shift.loc[_shift['last_price'].isnull() == True, 'w_price']

            _shift.loc[_shift['last_arr'].isnull() == True, 'last_arr'] \
                = _shift.loc[_shift['last_arr'].isnull() == True, 'arranged_cnt']

            _shift.loc[_shift['delta_price'].isnull() == True, 'delta_price'] = 0.

            # 计算销量
            lp = _shift['last_price']
            _delta_p = _shift['delta_price']
            la = _shift['last_arr']
            dp = _delta_p / lp
            ed_pred_arr = la * (1 - dp * ed_val)

            # 后处理
            ed_pred_arr[ed_pred_arr < 0] = 0
            sku_data = sku_data.join(pd.DataFrame(_shift['delta_price'], columns=['delta_price'])) \
                .join(pd.DataFrame(ed_pred_arr, columns=['ed_pred_arr'])) \
                .fillna(method='bfill') \
                .fillna(method='ffill')
            sku_data_list.append(sku_data)

        return pd.concat(sku_data_list)

    @staticmethod
    def get_price_statistic_features(df_total_raw):
        """
        价格弹性衍生特征
        :param df_total_raw: (含价格弹性特征的)原始数据
        :return: 价格弹性(价格、销量)衍生特征
        """
        df_total_raw = df_total_raw.sort_values('dt')
        sku_data_list = []
        sale_price_dic = {}

        for sku, sku_data in df_total_raw.groupby('sku_id'):
            dt_array = sku_data.dt.unique()
            dt_lenth = len(dt_array)

            arr_list = sku_data[sku_data.is_train == 1]['arranged_cnt'].tolist()
            price_list = sku_data[sku_data.is_train == 1]['w_price'].tolist()

            # 将数据前移20天
            arr_array = np.concatenate([[arr_list[0]] * 20, arr_list])
            price_array = np.concatenate([[price_list[0]] * 20, price_list])

            # 历史销量
            itvl_avg = arr_array.cumsum() / (pd.Series(arr_array).index + 1)
            # 历史价格
            itvl_price = price_array.cumsum() / (pd.Series(price_array).index + 1)
            sale_price_dic['itvl_pred'] = ['itvl_avg', 'itvl_price']

            # 历史销量指数均值
            itvl_ewm_avg = pd.Series(arr_array).ewm(span=4, min_periods=1).mean()
            # 历史价格指数均值
            itvl_ewm_price = pd.Series(price_array).ewm(span=4, min_periods=1).mean()
            sale_price_dic['itvl_ewm_pred'] = ['itvl_ewm_avg', 'itvl_ewm_price']

            # 滑动窗口统计值
            roll_arr_obj = pd.Series(arr_array).rolling(7, min_periods=1)
            roll_price_obj = pd.Series(price_array).rolling(7, min_periods=1)

            itvl_roll_avg = roll_arr_obj.mean()  # 销量滑动窗口均值
            itvl_roll_max = roll_arr_obj.max()  # 销量滑动窗口最大值
            itvl_roll_min = roll_arr_obj.min()  # 销量滑动窗口最小值

            itvl_roll_price_avg = roll_price_obj.mean()  # 价格滑动窗口均值差值
            itvl_roll_price_max = roll_price_obj.max()  # 价格滑动窗口最大值差值
            itvl_roll_price_min = roll_price_obj.min()  # 价格滑动窗口最小值差值

            sale_price_dic['itvl_roll_avg_pred'] = ['itvl_roll_avg', 'itvl_roll_price_avg']
            sale_price_dic['itvl_roll_max_pred'] = ['itvl_roll_max', 'itvl_roll_price_max']
            sale_price_dic['itvl_roll_min_pred'] = ['itvl_roll_min', 'itvl_roll_price_min']

            sku_data_list.append(pd.DataFrame({'sku_id': [sku] * dt_lenth,
                                               'itvl_avg': itvl_avg,
                                               'itvl_price': itvl_price,
                                               'itvl_ewm_avg': itvl_ewm_avg,
                                               'itvl_ewm_price': itvl_ewm_price,
                                               'itvl_roll_avg': itvl_roll_avg,
                                               'itvl_roll_price_avg': itvl_roll_price_avg,
                                               'itvl_roll_max': itvl_roll_max,
                                               'itvl_roll_price_max': itvl_roll_price_max,
                                               'itvl_roll_min': itvl_roll_min,
                                               'itvl_roll_price_min': itvl_roll_price_min,
                                               'dt': dt_array
                                               }))
        df_total_raw = df_total_raw.merge(pd.concat(sku_data_list), how='left', on=['sku_id', 'dt'])

        # 差价特征
        df_total_raw['itvl_avg_pdiff'] = df_total_raw['w_price'] - df_total_raw['itvl_price']
        df_total_raw['itvl_ewm_pdiff'] = df_total_raw['w_price'] - df_total_raw['itvl_ewm_price']
        df_total_raw['itvl_roll_avg_pdiff'] = df_total_raw['w_price'] - df_total_raw['itvl_roll_price_avg']
        df_total_raw['itvl_roll_max_pdiff'] = df_total_raw['w_price'] - df_total_raw['itvl_roll_price_max']
        df_total_raw['itvl_roll_min_pdiff'] = df_total_raw['w_price'] - df_total_raw['itvl_roll_price_min']

        # 价格相关销量
        ed_val = df_total_raw['ed_val']
        price = df_total_raw['w_price']
        for key, value in sale_price_dic.items():
            la = df_total_raw[value[0]]
            lp = df_total_raw[value[1]]
            _delta_p = price - lp
            dp = _delta_p / lp

            ed_pred_arr = la * (1 - dp * ed_val)
            ed_pred_arr[ed_pred_arr < 0] = 0

            df_total_raw = df_total_raw.join(pd.DataFrame(ed_pred_arr, columns=[key]))

        return df_total_raw.fillna(0.)

    @staticmethod
    def same_cat_rebate(data):
        """
        竞品折扣信息
        :param data: 原始数据
        :return: 竞品折扣
        """
        # 构造差价特征
        data['price_diff_seq'] = data['csu_origin_price'] - data['seq_price']
        data['price_diff_discount'] = data['csu_origin_price'] - data['discount_price']
        data['price_diff_sd'] = data['discount_price'] - data['seq_price']

        # 构造折扣特征
        data['seq_rebate'] = data['price_diff_seq'] / data['csu_origin_price']
        data['pro_rebate'] = data['price_diff_discount'] / data['csu_origin_price']

        # 秒杀折扣
        seq_data = data.groupby(['dt', 'cat2_name'], as_index=False)['seq_rebate'] \
            .agg({'max_seq_rebate': np.max, 'mean_seq_rebate': np.mean})

        # 大促折扣
        pro_data = data.groupby(['dt', 'cat2_name'], as_index=False)['pro_rebate'] \
            .agg({'max_pro_rebate': np.max, 'mean_pro_rebate': np.mean})

        # 合并数据
        rebate_data = seq_data.merge(pro_data, how='outer', on=['dt', 'cat2_name'])

        return data.merge(rebate_data, how='left', on=['dt', 'cat2_name']).fillna(0.)

    def his_sale_avg(self, raw_data):
        """
        移动平均销量特征
        """
        data = raw_data.sort_values(self.dt_field)
        sku_data_list = []

        for sku, sku_data in data.groupby('sku_id'):
            df_sku = sku_data[sku_data.is_train == 1]
            arr_list = df_sku['arranged_cnt']

            # 计算7天/15天/30天移动平均值
            roll_7d_avg = arr_list.rolling(7, min_periods=1).mean()
            roll_15d_avg = arr_list.rolling(15, min_periods=1).mean()
            roll_30d_avg = arr_list.rolling(30, min_periods=1).mean()

            # 计算(训练集)加权均值
            if self.cat1_id in [10021228]:
                his_avg = roll_7d_avg * 0.6 + roll_15d_avg * 0.3 + roll_30d_avg * 0.1
            else:
                his_avg = (roll_7d_avg + roll_15d_avg + roll_30d_avg) / 3.

            # 均值外推, 覆盖未来20天
            arr_and_avg = arr_list.tolist()
            arr_and_avg.append(his_avg.iloc[-1])
            for i in range(19):
                if len(arr_and_avg) < 7:
                    avg_7d = np.mean(arr_and_avg)
                else:
                    avg_7d = np.mean(arr_and_avg[-7:])

                if len(arr_and_avg) < 15:
                    avg_15d = np.mean(arr_and_avg)
                else:
                    avg_15d = np.mean(arr_and_avg[-15:])

                if len(arr_and_avg) < 30:
                    avg_30d = np.mean(arr_and_avg)
                else:
                    avg_30d = np.mean(arr_and_avg[-30:])
                if self.cat1_id in [10021228]:
                    arr_and_avg.append(avg_7d * 0.6 + avg_15d * 0.3 + avg_30d * 0.1)
                else:
                    arr_and_avg.append((avg_7d + avg_15d + avg_30d) / 3.)
            his_avg = np.concatenate([[his_avg.iloc[0]], his_avg, arr_and_avg[-19:]])

            try:
                sku_data_list.append(pd.DataFrame({'sku_id': sku,
                                                   'dt': sku_data.dt,
                                                   'his_avg': his_avg}))
            except:
                print("{} 历史销量特征制造异常!", str(sku))
                continue
        # 最终结果
        data = data.merge(pd.concat(sku_data_list), how='left', on=['sku_id', 'dt']).dropna()
        return data

    def his_pro_sale_avg(self, raw_data):
        """
        区分促销/非促销的历史销量特征
        :param raw_data: 原始数据
        :return: 历史销量
        """
        df_total = raw_data.sort_values(self.dt_field).reset_index(drop=True)

        # 区分促销/非促销数据
        # 非促销数据
        df_total_normal = df_total[(df_total['pro_num'] == 0) &
                                   (df_total['seq_num'] == 0) &
                                   (df_total['csu_redu_num'] == 0)]
        # 促销数据
        df_total_pro = df_total[~df_total.index.isin(df_total_normal.index)]

        sku_data_list = []
        for data in [df_total_normal, df_total_pro]:
            for sku, sku_data in data.groupby('sku_id'):
                df_sku = sku_data[sku_data.is_train == 1]

                # 训练集没有数据
                if len(df_sku) == 0:
                    continue

                arr_list = df_sku['arranged_cnt']
                days = len(sku_data[sku_data.is_train == 0]) - 1

                roll_7d_avg = arr_list.rolling(7, min_periods=1).mean()
                roll_15d_avg = arr_list.rolling(15, min_periods=1).mean()
                roll_30d_avg = arr_list.rolling(30, min_periods=1).mean()

                # 计算(训练集)加权均值
                his_avg = roll_7d_avg * 0.7 + roll_15d_avg * 0.2 + roll_30d_avg * 0.1

                # 均值外推, 覆盖测试集
                # 测试集没有数据
                if days == -1:
                    his_avg = np.concatenate([[his_avg.iloc[0]], his_avg[:-1]])
                # 测试集只有一天
                elif days == 0:
                    his_avg = np.concatenate([[his_avg.iloc[0]], his_avg])
                else:
                    arr_and_avg = arr_list.tolist()
                    arr_and_avg.append(his_avg.iloc[-1])

                    for i in range(days):
                        if len(arr_and_avg) < 7:
                            avg_7d = np.mean(arr_and_avg)
                        else:
                            avg_7d = np.mean(arr_and_avg[-7:])

                        if len(arr_and_avg) < 15:
                            avg_15d = np.mean(arr_and_avg)
                        else:
                            avg_15d = np.mean(arr_and_avg[-15:])

                        if len(arr_and_avg) < 30:
                            avg_30d = np.mean(arr_and_avg)
                        else:
                            avg_30d = np.mean(arr_and_avg[-30:])
                        arr_and_avg.append(avg_7d * 0.7 + avg_15d * 0.2 + avg_30d * 0.1)
                    his_avg = np.concatenate([[his_avg.iloc[0]], his_avg, arr_and_avg[-days:]])

                try:
                    sku_data_list.append(pd.DataFrame({'sku_id': sku,
                                                       'dt': sku_data.dt,
                                                       'pro_his_avg': his_avg}))
                except:
                    print("{} 历史(促销)销量特征制造异常!", str(sku))
                    continue
        final_data = df_total.merge(pd.concat(sku_data_list), how='left', on=['sku_id', 'dt'])

        # (训练集无数据导致的)空值处理
        final_data.loc[final_data['pro_his_avg'].isnull() == True, 'pro_his_avg'] \
            = final_data.loc[final_data['pro_his_avg'].isnull() == True, 'his_avg']
        return final_data

    def create_features(self, total_data, total_train_data):
        """
        特征制造函数
        :param total_data: 原始数据
        :param total_train_data: 训练数据
        :return: （合并后的）特征数据
        """
        all_data = []
        for wh_id, data in total_data.groupby('wh_id'):
            try:
                # 日期处理
                data['year'] = pd.to_datetime(data[self.dt_field]).dt.year
                data['month'] = pd.to_datetime(data[self.dt_field]).dt.month
                data['day'] = pd.to_datetime(data[self.dt_field]).dt.day

                train_data = total_train_data[total_train_data.wh_id == wh_id]
                train_data['month'] = pd.to_datetime(train_data[self.dt_field]).dt.month

                # 特征生产
                data = DerivedFeature.cnt_band_func(data)
                data = self.his_sale_avg(data)
                if self.cat1_id in [10021228]:
                    data = self.price_derived_features(data)
                    data = self.get_price_elasticity(data)
                    data = self.ed_pred_sale(data)
                    data = self.get_price_statistic_features(data)
                else:
                    data = self.his_pro_sale_avg(data)
                    data = self.same_cat_rebate(data)
                    data = self.get_statistic_features(data)
                    data = self.pro_statistic_features(data)
                data = self.seasonal_count(data, train_data)
                data = self.get_sale_ratio(data, train_data)
                byr_cols = [col for col in data.columns if '_byrs' in col]
                data = self.byrs_count(data, train_data, col_list=byr_cols)
                all_data.append(data)
            except:
                print('%s 仓 %s 品类特征制造异常！', str(wh_id), str(self.cat1_id))
                continue
        return pd.concat(all_data)
