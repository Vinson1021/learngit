import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

"""
量价模型-特征数据处理-工具类
"""

class FeatureUtil:

    def __init__(self):
        pass

    def sku_statistic_features(self, df_total_raw):
        """
        SKU销量统计特征
        """
        total_sku_data = []
        for sku in df_total_raw.sku_id.unique():
            # 将sku销售数据按天排序（从近到远）
            df_sku = df_total_raw[df_total_raw.sku_id == sku].sort_values('dt', ascending=False) \
                .reset_index(drop=True)
            df_sku['dt_before'] = df_sku.dt - timedelta(2)

            # 当前日期-1
            dt_before = df_sku.dt_before.iloc[0]

            # arr_data可能为空，填充“0.”后不影响结果
            arr_data = df_sku[df_sku.dt <= dt_before][['dt', 'arranged_cnt']]

            # 按日期从远到近排序, 应用窗口函数
            arr_data = arr_data.sort_values('dt', ascending=True).reset_index(drop=True)
            arr_data['dt_before'] = arr_data['dt']
            arr_list = arr_data.arranged_cnt.tolist()

            ## 历史均值
            itvl_avg = np.array(arr_list).cumsum() / (arr_data.index + 1.)

            ## 历史指数平滑均值
            itvl_ewm_avg = pd.Series(arr_list).ewm(span=5).mean()

            ## 滑动窗口对象
            roll_obj = pd.Series(arr_list).rolling(7, min_periods=1)

            ## 滑动窗口统计值
            itvl_roll_avg = roll_obj.mean()     ## 滑动窗口均值
            itvl_roll_max = roll_obj.max()      ## 滑动窗口最大值
            itvl_roll_mim = roll_obj.min()      ## 滑动窗口最小值

            ## 特征列赋值
            arr_data['itvl_avg'] = itvl_avg
            arr_data['itvl_ewm_avg'] = itvl_ewm_avg
            arr_data['itvl_roll_avg'] = itvl_roll_avg
            arr_data['itvl_roll_max'] = itvl_roll_max
            arr_data['itvl_roll_mim'] = itvl_roll_mim


            ## 合并数据，填补Nan值
            made_features = arr_data.drop(['dt', 'arranged_cnt'], axis=1).columns
            df_sku = df_sku.merge(arr_data[made_features], how='left', on=['dt_before'])
            df_sku = df_sku.fillna(method='bfill').fillna(0.)
            total_sku_data.append(df_sku.drop('dt_before', axis=1))

        return pd.concat(total_sku_data).sort_values('dt').reset_index(drop=True)


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


    def seasonal_count(self, df_total_raw, df_train_total):
        """
        季节性统计特征
        """

        # 使用全部训练数据计算品类销量均值

        ## 一级品类维度
        cat1_sale_mean = df_train_total.groupby('cat1_name', as_index=False).arranged_cnt.mean()
        ## 二级品类维度
        cat2_sale_mean = df_train_total.groupby('cat2_name', as_index=False).arranged_cnt.mean()

        # 一级品类星期销量均值/日常销量均值
        cat1_week_mean = df_train_total.groupby(['cat1_name','day_abbr'], as_index=False).arranged_cnt.mean()
        cat1_week_mean = cat1_week_mean.merge(cat1_sale_mean, how='left',
                                              on=['cat1_name'], suffixes=('_week', '_total'))
        cat1_week_mean['cat1_week_avg'] = cat1_week_mean['arranged_cnt_week'] / cat1_week_mean['arranged_cnt_total']
        df_total_raw = df_total_raw.merge(cat1_week_mean[['cat1_name','day_abbr', 'cat1_week_avg']],
                                          how='left', on=['cat1_name','day_abbr'])

        # 二级品类星期销量均值/日常销量均值
        cat2_week_mean = df_train_total.groupby(['cat2_name','day_abbr'], as_index=False).arranged_cnt.mean()
        cat2_week_mean = cat2_week_mean.merge(cat2_sale_mean, how='left',
                                              on=['cat2_name'], suffixes=('_week', '_total'))
        cat2_week_mean['cat2_week_avg'] = cat2_week_mean['arranged_cnt_week'] / cat2_week_mean['arranged_cnt_total']
        df_total_raw = df_total_raw.merge(cat2_week_mean[['cat2_name','day_abbr', 'cat2_week_avg']],
                                          how='left', on=['cat2_name','day_abbr'])

        # 一级品类月份销量均值/日常销量均值
        cat1_month_mean = df_train_total.groupby(['cat1_name','month'], as_index=False).arranged_cnt.mean()
        cat1_month_mean = cat1_month_mean.merge(cat1_sale_mean, how='left',
                                                on=['cat1_name'], suffixes=('_week', '_total'))
        cat1_month_mean['cat1_month_avg'] = cat1_month_mean['arranged_cnt_week'] / cat1_month_mean['arranged_cnt_total']
        df_total_raw = df_total_raw.merge(cat1_month_mean[['cat1_name','month', 'cat1_month_avg']],
                                          how='left', on=['cat1_name','month'])

        # 二级品类月份销量均值/日常销量均值
        cat2_month_mean = df_train_total.groupby(['cat2_name','month'], as_index=False).arranged_cnt.mean()
        cat2_month_mean = cat2_month_mean.merge(cat2_sale_mean, how='left',
                                                on=['cat2_name'], suffixes=('_week', '_total'))
        cat2_month_mean['cat2_month_avg'] = cat2_month_mean['arranged_cnt_week'] / cat2_month_mean['arranged_cnt_total']
        df_total_raw = df_total_raw.merge(cat2_month_mean[['cat2_name','month', 'cat2_month_avg']],
                                          how='left', on=['cat2_name','month'])

        # 一级品类节假日销量均值/日常销量均值
        cat1_fevl_mean = df_train_total.groupby(['cat1_name','festival_name'], as_index=False).arranged_cnt.mean()
        cat1_fevl_mean.rename(columns={'arranged_cnt': 'cat1_fevl_avg'}, inplace=True)
        cat1_fevl_mean = cat1_fevl_mean.merge(cat1_sale_mean)
        cat1_fevl_mean['cat1_fevl_avg'] = cat1_fevl_mean.cat1_fevl_avg / cat1_fevl_mean.arranged_cnt
        del cat1_fevl_mean['arranged_cnt']
        df_total_raw= df_total_raw.merge(cat1_fevl_mean, how='left', on=['cat1_name','festival_name'])

        # 二级品类节假日销量均值/日常销量均值
        cat2_fevl_mean = df_train_total.groupby(['cat2_name','festival_name'], as_index=False).arranged_cnt.mean()
        cat2_fevl_mean.rename(columns={'arranged_cnt': 'cat2_fevl_avg'}, inplace=True)
        cat2_fevl_mean = cat2_fevl_mean.merge(cat2_sale_mean)
        cat2_fevl_mean['cat2_fevl_avg'] = cat2_fevl_mean.cat2_fevl_avg / cat2_fevl_mean.arranged_cnt
        del cat2_fevl_mean['arranged_cnt']
        df_total_raw= df_total_raw.merge(cat2_fevl_mean, how='left', on=['cat2_name','festival_name'])

        # 一级品类西方节假日销量均值/日常销量均值
        cat1_wfevl_mean = df_train_total.groupby(['cat1_name','western_festival_name'], as_index=False).arranged_cnt.mean()
        cat1_wfevl_mean.rename(columns={'arranged_cnt': 'cat1_wfevl_avg'}, inplace=True)
        cat1_wfevl_mean = cat1_wfevl_mean.merge(cat1_sale_mean)
        cat1_wfevl_mean['cat1_wfevl_avg'] = cat1_wfevl_mean.cat1_wfevl_avg / cat1_wfevl_mean.arranged_cnt
        del cat1_wfevl_mean['arranged_cnt']
        df_total_raw= df_total_raw.merge(cat1_wfevl_mean, how='left', on=['cat1_name','western_festival_name'])

        # 二级品类西方节假日销量均值/日常销量均值
        cat2_wfevl_mean = df_train_total.groupby(['cat2_name','western_festival_name'], as_index=False).arranged_cnt.mean()
        cat2_wfevl_mean.rename(columns={'arranged_cnt': 'cat2_wfevl_avg'}, inplace=True)
        cat2_wfevl_mean = cat2_wfevl_mean.merge(cat2_sale_mean)
        cat2_wfevl_mean['cat2_wfevl_avg'] = cat2_wfevl_mean.cat2_wfevl_avg / cat2_wfevl_mean.arranged_cnt
        del cat2_wfevl_mean['arranged_cnt']
        df_total_raw= df_total_raw.merge(cat2_wfevl_mean, how='left', on=['cat2_name','western_festival_name'])

        return df_total_raw.fillna(0.)


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

            # 使用全部训练数据计算商户数均值
            _mean = df_train_total.groupby('cat1_name', as_index=False)[[col]].mean()

            # 品类星期商户均值/日常商户均值
            _week_mean = df_train_total.groupby(['cat1_name', 'day_abbr'], as_index=False)[[col]].mean()
            _week_mean = _week_mean.merge(_mean, how='left', on=['cat1_name'])
            _week_mean[week_col_name] = _week_mean.iloc[:, -2] / _week_mean.iloc[:, -1]

            # 品类月份商户均值/日常商户均值
            _month_mean = df_train_total.groupby(['cat1_name', 'month'], as_index=False)[[col]].mean()
            _month_mean = _month_mean.merge(_mean, how='left', on=['cat1_name'])
            _month_mean[month_col_name] = _month_mean.iloc[:, -2] / _month_mean.iloc[:, -1]

            # 品类节假日商户均值/日常商户均值
            _fevl_mean = df_train_total.groupby(['cat1_name','festival_name'], as_index=False)[[col]].mean()
            _fevl_mean = _fevl_mean.merge(_mean, how='left', on=['cat1_name'])
            _fevl_mean[fevl_col_name] = _fevl_mean.iloc[:, -2] / _fevl_mean.iloc[:, -1]

            # 品类西方节假日商户均值/日常商户均值
            _wfevl_mean = df_train_total.groupby(['cat1_name','western_festival_name'], as_index=False)[[col]].mean()
            _wfevl_mean = _wfevl_mean.merge(_mean, how='left', on=['cat1_name'])
            _wfevl_mean[wfevl_col_name] = _wfevl_mean.iloc[:, -2] / _wfevl_mean.iloc[:, -1]

            df_total_raw = df_total_raw.merge(_week_mean[['cat1_name','day_abbr', week_col_name]],
                                              how='left', on=['cat1_name','day_abbr'])
            df_total_raw = df_total_raw.merge(_month_mean[['cat1_name','month', month_col_name]],
                                              how='left', on=['cat1_name','month'])
            df_total_raw = df_total_raw.merge(_fevl_mean[['cat1_name','festival_name', fevl_col_name]],
                                              how='left', on=['cat1_name','festival_name'])
            df_total_raw = df_total_raw.merge(_wfevl_mean[['cat1_name','western_festival_name', wfevl_col_name]],
                                              how='left', on=['cat1_name','western_festival_name'])

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
                cnt_label = pd.cut(group.arr_mean, cut_num, precision=2, labels=label_names)
                df_label = pd.DataFrame({'dt': dt_key,
                                         'cat2_name': cat2,
                                         'sku_id': group.sku_id,
                                         'cnt_band_area': cnt_label.astype(np.int32)
                                         })
                cnt_band_data.append(df_label)

            ## 计算全品类下各sku销量band
            qpl_cnt_label = pd.cut(qpl_group.arr_mean, cut_num, precision=2, labels=label_names)
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


    def price_derived_features(self, df_total_raw):
        """
        价格衍生特征
        """
        # 活动差价
        df_total_raw['pro_seq_pdiff'] = df_total_raw['pro_price'] - df_total_raw['seq_price']

        # 折扣力度
        price_diff_seq = df_total_raw['price'] - df_total_raw['seq_price']
        price_diff_pro = df_total_raw['price'] - df_total_raw['pro_price']
        df_total_raw['seq_rebate'] = price_diff_seq / df_total_raw['price']
        df_total_raw['pro_rebate'] = price_diff_pro / df_total_raw['price']

        # 竞品折扣
        ## 竞品秒杀力度
        seq_data = df_total_raw.groupby(['dt','cat2_name'], as_index=False)['seq_rebate'] \
            .agg({'max_seq_rebate': np.max,
                  'mean_seq_rebate': np.mean})
        ## 竞品单品促销力度
        pro_data = df_total_raw.groupby(['dt','cat2_name'], as_index=False)['pro_rebate'] \
            .agg({'max_pro_rebate': np.max,
                  'mean_pro_rebate': np.mean})

        # 合并数据
        rebate_data = seq_data.merge(pro_data, how='outer', on=['dt', 'cat2_name'])
        df_total_raw = df_total_raw.merge(rebate_data, how='left', on=['dt','cat2_name']) \
            .fillna(0.) \
            .sort_values('dt')

        # 滑动价格特征
        sku_data_list = []
        for sku, sku_data in df_total_raw.groupby('sku_id'):
            df_sku = sku_data[sku_data.is_train == 1]

            ## 14天滑动价格均值
            price_14d_roll = df_sku['price'].rolling(14, min_periods=1).mean()
            mov_price_avg = np.insert(np.array(price_14d_roll), 0, price_14d_roll.iloc[0], axis=0)

            ## 构造T+1数据
            if len(mov_price_avg) < 14:
                price_avg = np.mean(mov_price_avg)
            else:
                price_avg = np.mean(mov_price_avg[-14:])

            mov_price_avg = np.append(mov_price_avg, price_avg)

            sku_data_list.append(pd.DataFrame({'sku_id': sku,
                                               'dt': sku_data.dt.unique(),
                                               'mov_price_avg': mov_price_avg
                                               }))
        ## 合并sku数据
        df_total_raw = df_total_raw.merge(pd.concat(sku_data_list), how='left', on=['sku_id', 'dt'])
        ## 原价/滑动价格差价
        df_total_raw['roll_price_diff'] = df_total_raw['price'] - df_total_raw['mov_price_avg']


        # 价格一阶差分
        train_data = df_total_raw[df_total_raw.is_train == 1].sort_values(['sku_id', 'dt'])
        test_data = df_total_raw[df_total_raw.is_train == 0]

        ## 训练集价格差分
        train_data['price_fdiff'] = train_data['price'].diff()

        ## 各sku首条记录数据修正
        zero_index = []       ### 要修正的数据索引
        sku_price = []        ### 各sku最近记录的价格
        for sku, sku_data in train_data.groupby('sku_id'):
            zero_index.append(sku_data.index[0])
            sku_price.append([sku, sku_data.price.iloc[-1]])
        train_data.loc[zero_index, 'price_fdiff'] = 0.

        ## 测试集价格差分
        sku_price = pd.DataFrame(sku_price, columns=['sku_id', 'last_price'])
        test_temp = test_data.merge(sku_price, how='left', on=['sku_id'])
        test_data['price_fdiff'] = np.array(test_temp['price'] - test_temp['last_price'])

        return pd.concat([train_data, test_data])


    def price_elasticity(self, df_total_raw):
        """
        价格弹性系数
        """
        df_total_raw = df_total_raw.sort_values('dt')

        # 当日决定价格
        df_total_raw['final_price'] = df_total_raw[['price', 'seq_price', 'pro_price']].min(axis=1)

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
            price_and_arr = sku_train[['final_price', 'arranged_cnt']].rolling(15, min_periods=15).mean()

            ## 价格和销量变动值
            _delta = (price_and_arr - price_and_arr.shift(1)).rename(columns={'final_price':'delta_price',
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
            p = _delta['final_price']
            ed = -(darr / arr) / (dp / p)

            ## 过滤弹性系数中的异常值
            Q1 = np.percentile(ed, 25)
            Q3 = np.percentile(ed, 75)
            step = (Q3 - Q1) * 1.5
            mean_ed = ed[(ed <= Q3 + step) & (ed >= Q1 - step)].mean()

            ## 更新弹性系数
            ed_dic[sku] = mean_ed

        # 循环2: 制造弹性价格相关销量
        for sku, sku_data in df_total_raw.groupby('sku_id'):

            ## 获取弹性系数
            mean_ed = ed_dic[sku]

            ## 用上一日数据计算差值
            _shift = sku_data[['final_price', 'arranged_cnt']].shift(1) \
                .rename(columns={'final_price':'last_price',
                                 'arranged_cnt':'last_arr'})
            _shift = _shift.join(sku_data[['final_price','arranged_cnt','is_train']])

            ## 修正测试集数据
            last_price = sku_data[sku_data.is_train == 1].final_price.iloc[-1]
            last_arr = sku_data[sku_data.is_train == 1].arranged_cnt.iloc[-1]
            _shift.loc[_shift.is_train == 0, 'last_price'] = last_price
            _shift.loc[_shift.is_train == 0, 'last_arr'] = last_arr

            ## 计算相关销量
            fp = _shift['final_price']
            lp = _shift['last_price']
            la = _shift['last_arr']
            dp = (fp - lp) / lp
            ed_pred_arr = la * (1 - dp * mean_ed)

            ## 后处理
            ed_pred_arr[ed_pred_arr < 0] = 0
            sku_data = sku_data.join(pd.DataFrame(ed_pred_arr, columns=['ed_pred_arr'])) \
                .fillna(method='bfill') \
                .fillna(method='ffill')
            sku_data_list.append(sku_data)

        return pd.concat(sku_data_list)



