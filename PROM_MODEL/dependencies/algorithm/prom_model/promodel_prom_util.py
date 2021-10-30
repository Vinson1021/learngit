#!/bin/usr/env python3.6
# -*- coding: utf-8 -*-
# ---------------------------------
# ProjectName: data-forecast-spark
# Description: 促销模型促销特征
# Author: guoyunshen
# CreateTime: 2020-10-21 11:12
# ---------------------------------

import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class PriceUtil:

    def __init__(self):
        pass


    def price_derived_features(self, df_total_raw):
        """
        价格衍生特征
        """
        # 促销活动差价
        df_total_raw['sp_pdiff'] = df_total_raw['seq_price'] - df_total_raw['pro_price']

        # 真实差价
        df_total_raw['real_pdiff'] = df_total_raw['price'] - df_total_raw['w_price']

        # 折扣力度
        price_diff_seq = df_total_raw['price'] - df_total_raw['seq_price']
        price_diff_pro = df_total_raw['price'] - df_total_raw['pro_price']
        price_diff_real = df_total_raw['price'] - df_total_raw['w_price']

        ## 秒杀折扣
        df_total_raw['seq_rebate'] = price_diff_seq / df_total_raw['price']
        ## 促销折扣
        df_total_raw['pro_rebate'] = price_diff_pro / df_total_raw['price']
        ## 实际折扣
        df_total_raw['real_rebate'] = price_diff_real / df_total_raw['price']


        # 竞品折扣
        ## 竞品秒杀力度
        seq_data = df_total_raw.groupby(['dt','cat2_name'], as_index=False)['seq_rebate'] \
            .agg({'max_seq_rebate': np.max,
                  'mean_seq_rebate': np.mean})
        ## 竞品单品促销力度
        pro_data = df_total_raw.groupby(['dt','cat2_name'], as_index=False)['pro_rebate'] \
            .agg({'max_pro_rebate': np.max,
                  'mean_pro_rebate': np.mean})
        ## 竞品实际折扣
        real_data = df_total_raw.groupby(['dt','cat2_name'], as_index=False)['real_rebate'] \
            .agg({'max_real_rebate': np.max,
                  'mean_real_rebate': np.mean})

        # 合并数据
        rebate_data = seq_data.merge(pro_data, how='outer', on=['dt', 'cat2_name'])
        rebate_data = rebate_data.merge(real_data, how='outer', on=['dt', 'cat2_name'])
        df_total_raw = df_total_raw.merge(rebate_data, how='left', on=['dt','cat2_name']) \
            .fillna(0.) \
            .sort_values('dt')

        # 滑动价格
        sku_data_list = []
        for sku, sku_data in df_total_raw.groupby('sku_id'):
            df_sku = sku_data[sku_data.is_train == 1]

            ## 7天(加权)价格滑动均值
            price_7d_roll = df_sku['w_price'].rolling(7, min_periods=1).mean()

            ## T+0数据
            mov_price_avg = np.insert(np.array(price_7d_roll), 0, price_7d_roll.iloc[0], axis=0)

            ## 构造T+1数据
            price_list = np.append(df_sku['w_price'], price_7d_roll.iloc[-1])
            if len(price_list) < 7:
                price_avg = np.mean(price_list)
            else:
                price_avg = np.mean(price_list[-7:])

            mov_price_avg = np.append(mov_price_avg, price_avg)

            sku_data_list.append(pd.DataFrame({'sku_id': sku,
                                               'dt': sku_data.dt.unique(),
                                               'mov_price_avg': mov_price_avg
                                               }))
        ## 合并sku数据
        df_total_raw = df_total_raw.merge(pd.concat(sku_data_list), how='left', on=['sku_id', 'dt'])

        ## 原价/滑动价格差价
        df_total_raw['mov_delta_price'] = df_total_raw['price'] - df_total_raw['mov_price_avg']

        return df_total_raw


    def get_price_elasticity(self, df_total_raw):
        """
        价格弹性系数
        """
        df_total_raw = df_total_raw.sort_values('dt')

        # 返回结果
        sku_data_list = []

        # 赋初始值
        ed_dic = {sku: 0. for sku in df_total_raw.sku_id.unique()}

        # 过滤满减数据
        df_ed = df_total_raw[(df_total_raw.is_train == 1) &
                             (df_total_raw.redu_num == 0)]

        # 计算价格弹性系数
        for sku, sku_train in df_ed.groupby('sku_id'):

            ## 数据量限制
            if len(sku_train) < 7:
                continue

            ## 使用滑动均值价格
            price_and_arr = sku_train[['w_price', 'arranged_cnt']].rolling(7, min_periods=7).mean()

            ## 价格和销量变动值
            _delta = (price_and_arr - price_and_arr.shift(1)).rename(columns={'w_price':'delta_price',
                                                                              'arranged_cnt':'delta_arr'})
            ### 合并两部分数据
            _delta = price_and_arr.join(_delta.shift(-1)).dropna()
            _delta = _delta[_delta.delta_price != 0]

            ## 过滤价格无波动的sku
            if len(_delta) == 0:
                continue

            ## 计算价格弹性系数
            darr = _delta['delta_arr']
            arr = _delta['arranged_cnt']
            dp = _delta['delta_price']
            p = _delta['w_price']
            ed = -(darr / arr) / (dp / p)

            ## 过滤弹性系数中的异常值
            Q1 = np.percentile(ed, 25)
            Q3 = np.percentile(ed, 75)
            step = (Q3 - Q1) * 1.5
            mean_ed = ed[(ed <= Q3 + step) & (ed >= Q1 - step)].mean()

            ## 更新弹性系数
            ed_dic[sku] = mean_ed

        # 生成价格弹性特征
        ed_list = [[sku_id, ed_dic[sku_id]] for sku_id in ed_dic.keys()]
        df_ed = pd.DataFrame(ed_list, columns=['sku_id', 'ed_val'])

        # 总体过滤异常值
        t_Q1 = np.percentile(df_ed.ed_val, 25)
        t_Q3 = np.percentile(df_ed.ed_val, 75)
        t_step = (t_Q3 - t_Q1) * 1.5
        top = t_Q3 + t_step
        bottom = t_Q1 - t_step
        df_ed.loc[(df_ed.ed_val > top), 'ed_val'] = top
        df_ed.loc[(df_ed.ed_val < bottom), 'ed_val'] = bottom

        # 负价格弹性替换
        df_total_raw = df_total_raw.merge(df_ed, how='left', on=['sku_id'])
        df_cat_ed = df_total_raw.groupby('cat2_name', as_index=False)['ed_val'] \
            .agg({'cat2_ed_avg': np.mean})
        df_total_raw = df_total_raw.merge(df_cat_ed, how='left', on=['cat2_name']) \
            .sort_values('dt') \
            .fillna(0.)
        df_total_raw.loc[df_total_raw.ed_val <= 0, 'ed_val'] \
            = df_total_raw.loc[df_total_raw.ed_val <= 0, 'cat2_ed_avg']

        return df_total_raw


    def non_linear_elasticity(self, df_total_raw):
        """
        非线性价格弹性
        """
        df_total_raw = df_total_raw.sort_values('dt')

        # 赋初始值
        ed_dic = {sku: [0., 0., 0.] for sku in df_total_raw.sku_id.unique()}

        # 过滤满减数据
        df_ed = df_total_raw[(df_total_raw.is_train == 1) &
                             (df_total_raw.redu_num == 0)]


        # 计算价格弹性系数
        for sku, sku_train in df_ed.groupby('sku_id'):

            ## 数据量限制
            if len(sku_train) < 15:
                continue

            ## 提取价格和销量的对应关系
            price_and_arr = sku_train[['w_price', 'arranged_cnt']].rolling(30, min_periods=7).mean()
            price_and_arr = price_and_arr.dropna()

            ## 价格无波动sku剔除
            if round(price_and_arr['w_price'].max(), 3) == round(price_and_arr['w_price'].min(), 3):
                continue

            ## GMV计算，加入了price的因子
            gmv = pow(price_and_arr['w_price'], 2) * price_and_arr['arranged_cnt']
            price_and_arr = price_and_arr.join(pd.DataFrame(gmv, columns=['gmv']))

            ## 过滤异常GMV值
            Q1 = np.percentile(price_and_arr['gmv'], 25)
            Q3 = np.percentile(price_and_arr['gmv'], 75)
            step = (Q3 - Q1)
            price_and_arr['gmv_flag'] = price_and_arr.apply(lambda x: 1 if x['gmv'] <= Q3 + step \
                                                                           and x['gmv'] >= Q1 - step else 0, axis=1)
            price_and_arr = price_and_arr[price_and_arr['gmv_flag'] == 1]

            ## 价格二次平滑
            price_and_arr = price_and_arr.sort_values('w_price')
            cnt = 0
            flag = [0]
            for i in range(1, price_and_arr.shape[0]):
                if i % 5 == 0:
                    cnt += 1
                flag.append(cnt)
            price_and_arr['price_band'] = flag
            price_and_arr = price_and_arr.groupby('price_band',as_index=False).mean()

            ## 用二次多项式拟合曲线
            pri = price_and_arr['w_price']
            arr = price_and_arr['arranged_cnt']
            curve = np.polyfit(pri, arr, 2)
            poly = np.poly1d(curve)

            ## 更新弹性系数
            ed_dic[sku] = poly

        df_total_raw['ed_val2'] = df_total_raw.apply(lambda row: -ed_dic[row['sku_id']][2], axis=1)
        df_total_raw['ed_val1'] = df_total_raw.apply(lambda row: -ed_dic[row['sku_id']][1], axis=1)
        df_total_raw['sec_ed'] = 2 * df_total_raw['w_price'] * df_total_raw['ed_val2'] + df_total_raw['ed_val1']

        ## 根据二级品类修正异常的sec_ed
        cat_data_list = []
        for cat, cat_data in df_total_raw.groupby('cat2_name'):
            ed = cat_data['sec_ed'].copy()
            Q1 = np.percentile(ed, 25)
            Q3 = np.percentile(ed, 75)
            step = (Q3 - Q1) * 1.5
            ed[ed > Q3 + step] = Q3 + step
            ed[ed < Q1 - step] = Q1 - step

            if not np.isnan(ed[ed > 0].mean()):
                _mean = ed[ed > 0].mean()
            else:
                _mean = 1.

            cat_data['sec_ed'] = ed
            cat_data['sec_ed'] = cat_data['sec_ed'].apply(lambda x: x if x > 0 else _mean)
            cat_data_list.append(cat_data)

        return pd.concat(cat_data_list)


    def self_ed_func(self, ed):
        """
        自定义价格弹性修正
        """
        if not np.isnan(ed[ed > 0].mean()):
            return ed[ed > 0].mean()
        else:
            return 1


    def get_price_ratio_elasticity(self, df_total_raw):
        """
        根据价格变化率的弹性系数
        """
        df_total_raw = df_total_raw.sort_values('dt')

        # 返回结果
        sku_data_list = []

        # 赋初始值
        ed_dic = {sku: 0. for sku in df_total_raw.sku_id.unique()}

        # 过滤满减数据
        df_ed = df_total_raw[(df_total_raw.is_train == 1) &
                             (df_total_raw.redu_num == 0)]

        # 循环1: 计算价格弹性系数
        for sku, sku_train in df_ed.groupby('sku_id'):

            ## 数据量限制
            if len(sku_train) < 15:
                continue

            ## 使用滑动均值价格
            price_and_arr = sku_train[['w_price', 'arranged_cnt']].rolling(7, min_periods=7).mean()

            # 价格和销量变动值
            _delta = (price_and_arr - price_and_arr.shift(1)).rename(columns={'w_price':'delta_price',
                                                                              'arranged_cnt':'delta_arr'})
            ### 合并两部分数据
            _delta = price_and_arr.join(_delta.shift(-1)).dropna()
            _delta = _delta[_delta.delta_price != 0]

            ## 过滤价格无波动的sku
            if len(_delta) == 0:
                continue

            delta_price_ratio = _delta['delta_price'] / _delta['w_price']
            delta_arr_ratio = _delta['delta_arr'] / _delta['arranged_cnt']
            dprice_and_darr = pd.DataFrame({'delta_price_ratio': delta_price_ratio,
                                            'delta_arr_ratio': delta_arr_ratio})

            ## 过滤弹性系数中的异常值
            Q1 = np.percentile(dprice_and_darr['delta_price_ratio'], 25)
            Q3 = np.percentile(dprice_and_darr['delta_price_ratio'], 75)
            step = (Q3 - Q1) * 1.5
            dprice_and_darr = dprice_and_darr[~((dprice_and_darr['delta_price_ratio'] > Q3 + step) |
                                                (dprice_and_darr['delta_price_ratio'] < Q1 - step))]

            ## 价格弹性计算
            dprice_and_darr = dprice_and_darr.sort_values('delta_price_ratio')
            cnt = 0
            flag = [0]
            for i in range(1,dprice_and_darr.shape[0]):
                if i % 5 == 0:
                    cnt += 1
                flag.append(cnt)

            dprice_and_darr['price_band'] = flag
            dprice_and_darr = dprice_and_darr.groupby('price_band',as_index=False).mean()

            pri = dprice_and_darr['delta_price_ratio']
            arr = dprice_and_darr['delta_arr_ratio']

            curve = np.polyfit(pri,arr, 1)
            poly = np.poly1d(curve)
            ed_dic[sku] = -poly[1]

        # 生成价格弹性特征
        ed_list = [[sku_id, ed_dic[sku_id]] for sku_id in ed_dic.keys()]
        df_ed = pd.DataFrame(ed_list, columns=['sku_id', 'ratio_ed_val'])

        # 总体过滤异常值
        t_Q1 = np.percentile(df_ed.ratio_ed_val, 25)
        t_Q3 = np.percentile(df_ed.ratio_ed_val, 75)
        t_step = (t_Q3 - t_Q1) * 1.5
        top = t_Q3 + t_step
        bottom = t_Q1 - t_step
        df_ed.loc[(df_ed.ratio_ed_val > top), 'ratio_ed_val'] = top
        df_ed.loc[(df_ed.ratio_ed_val < bottom), 'ratio_ed_val'] = bottom

        # 负价格弹性替换
        df_total_raw = df_total_raw.merge(df_ed, how='left', on=['sku_id'])
        df_cat_ed = df_total_raw.groupby('cat2_name', as_index=False)['ratio_ed_val'].agg([self.self_ed_func])
        df_cat_ed.rename(columns={'self_ed_func': 'cat2_ratio_ed_avg'}, inplace=True)
        df_total_raw = df_total_raw.merge(df_cat_ed, how='left', on=['cat2_name']) \
            .sort_values('dt') \
            .fillna(0.)
        df_total_raw.loc[df_total_raw.ratio_ed_val <= 0, 'ratio_ed_val'] \
            = df_total_raw.loc[df_total_raw.ratio_ed_val <= 0, 'cat2_ratio_ed_avg']

        return df_total_raw


    def ed_pred_sale(self, df_total_raw):
        """
        使用价格弹性计算销量
        """
        df_total_raw = df_total_raw.sort_values('dt')
        sku_data_list = []

        for sku, sku_data in df_total_raw.groupby('sku_id'):

            # 获取(一阶)弹性系数
            ed_val = sku_data['ed_val'].unique()[0]
            ratio_ed_val = sku_data['ratio_ed_val'].unique()[0]

            # 获取(二阶)价格弹性
            mean_ed = sku_data['sec_ed'].mean()
            ed = sku_data['sec_ed']

            _shift = sku_data[['w_price', 'arranged_cnt']].shift(1) \
                .rename(columns={'w_price':'last_price',
                                 'arranged_cnt':'last_arr'})
            _shift = _shift.join(sku_data[['w_price','arranged_cnt','is_train']])

            # 修正测试集数据
            last_price = sku_data[sku_data.is_train == 1]['w_price'].iloc[-1]
            last_arr = sku_data[sku_data.is_train == 1]['arranged_cnt'].iloc[-1]
            _shift.loc[_shift.is_train == 0, 'last_price'] = last_price
            _shift.loc[_shift.is_train == 0, 'last_arr'] = last_arr

            # 价格一阶差分
            _shift['delta_price'] = _shift['w_price'] - _shift['last_price']

            # 填补缺失值
            _shift.loc[_shift['last_price'].isnull()==True, 'last_price'] \
                = _shift.loc[_shift['last_price'].isnull()==True, 'w_price']

            _shift.loc[_shift['last_arr'].isnull()==True, 'last_arr'] \
                = _shift.loc[_shift['last_arr'].isnull()==True, 'arranged_cnt']

            _shift.loc[_shift['delta_price'].isnull()==True, 'delta_price'] = 0.

            # 计算(一阶)弹性销量
            lp = _shift['last_price']
            _delta_p = _shift['delta_price']
            la = _shift['last_arr']
            dp = _delta_p / lp
            ed_pred_arr = la * (1 - dp * ed_val)

            # 计算(二阶)弹性销量
            ed_pred_arr_nlm = la * (1 - dp * mean_ed)
            ed_pred_arr_nl = la * (1 - dp * ed)

            # 计算(一阶)变化率弹性销量
            ratio_ed_pred_arr = la * (1 - dp * ratio_ed_val)

            # 后处理
            ed_pred_arr[ed_pred_arr < 0] = 0
            sku_data = sku_data.join(pd.DataFrame(_shift['delta_price'], columns=['delta_price'])) \
                .join(pd.DataFrame(ed_pred_arr, columns=['ed_pred_arr'])) \
                .fillna(method='bfill') \
                .fillna(method='ffill')

            ed_pred_arr_nlm[ed_pred_arr_nlm < 0] = 0
            sku_data = sku_data.join(pd.DataFrame(ed_pred_arr_nlm, columns=['ed_pred_arr_nlm'])) \
                .fillna(method='bfill') \
                .fillna(method='ffill')

            ed_pred_arr_nl[ed_pred_arr_nl < 0] = 0
            sku_data = sku_data.join(pd.DataFrame(ed_pred_arr_nl, columns=['ed_pred_arr_nl'])) \
                .fillna(method='bfill') \
                .fillna(method='ffill')

            ratio_ed_pred_arr[ratio_ed_pred_arr < 0] = 0
            sku_data = sku_data.join(pd.DataFrame(ratio_ed_pred_arr, columns=['ratio_ed_pred_arr'])) \
                .fillna(method='bfill') \
                .fillna(method='ffill')

            sku_data_list.append(sku_data)

        return pd.concat(sku_data_list)


    def get_price_statistic_features(self, df_total_raw):
        """
        价格弹性衍生特征
        """
        df_total_raw = df_total_raw.sort_values('dt')
        sku_data_list = []
        sale_price_dic = {}

        begin_time = time.time()

        for sku, sku_data in df_total_raw.groupby('sku_id'):

            ed_val = sku_data.ed_val.unique()[0]
            dt_array = sku_data.dt.unique()
            dt_lenth = len(dt_array)

            arr_list = sku_data[sku_data.is_train == 1]['arranged_cnt'].tolist()
            price_list = sku_data[sku_data.is_train == 1]['w_price'].tolist()

            # 将数据前移2天
            arr_array = np.concatenate([[arr_list[0]] * 2, arr_list])
            price_array = np.concatenate([[price_list[0]] * 2, price_list])

            # 历史销量
            itvl_avg = arr_array.cumsum() / (pd.Series(arr_array).index + 1)
            # 历史价格
            itvl_price = price_array.cumsum() / (pd.Series(price_array).index + 1)
            sale_price_dic['itvl_pred'] = ['itvl_avg', 'itvl_price']

            # 历史销量指数均值
            itvl_ewm_avg = pd.Series(arr_array).ewm(span=7, min_periods=1).mean()
            # 历史价格指数均值
            itvl_ewm_price = pd.Series(price_array).ewm(span=7, min_periods=1).mean()
            sale_price_dic['itvl_ewm_pred'] = ['itvl_ewm_avg', 'itvl_ewm_price']

            # 滑动窗口统计值
            roll_arr_obj = pd.Series(arr_array).rolling(14, min_periods=1)
            roll_price_obj = pd.Series(price_array).rolling(14, min_periods=1)

            itvl_roll_avg = roll_arr_obj.mean()     ## 销量滑动窗口均值
            itvl_roll_max = roll_arr_obj.max()      ## 销量滑动窗口最大值
            itvl_roll_min = roll_arr_obj.min()      ## 销量滑动窗口最小值

            itvl_roll_price_avg = roll_price_obj.mean()     ## 价格滑动窗口均值差值
            itvl_roll_price_max = roll_price_obj.max()      ## 价格滑动窗口最大值差值
            itvl_roll_price_min = roll_price_obj.min()      ## 价格滑动窗口最小值差值

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
        col_list = ['sku_id', 'dt', 'arranged_cnt', 'w_price', 'ed_val']

        temp = df_total_raw[col_list]
        temp.drop_duplicates(subset=['sku_id', 'dt', 'w_price'],keep='first',inplace=True)
        temp = temp.merge(pd.concat(sku_data_list), how='left', on=['sku_id', 'dt'])

        # 预估价格相关销量
        ed_val = temp['ed_val']
        price = temp['w_price']
        for key, value in sale_price_dic.items():
            la = temp[value[0]]
            lp = temp[value[1]]
            _delta_p = price - lp
            dp = _delta_p / lp

            ed_pred_arr = la * (1 - dp * ed_val)
            ed_pred_arr[ed_pred_arr < 0] = 0

            temp = temp.join(pd.DataFrame(ed_pred_arr, columns=[key]))

        return df_total_raw.merge(temp, how='left', on=col_list)