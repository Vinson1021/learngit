import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json, traceback
import random
random.seed(42)
np.random.seed(42)
raw_feat_cols = 'is_train,wh_id,sku_id,cat1_name,cat2_name,cnt_band,brand_name,tax_rate,csu_origin_price,seq_num,seq_price,redu_num,pro_num,discount_price,day_abbr,is_work_day,is_weekend,is_holiday,festival_name,western_festival_name,weather,weather_b,avg_t,avg_t_b,new_byrs,old_byrs,poultry_byrs,vege_fruit_byrs,egg_aquatic_byrs,standard_byrs,vege_byrs,meet_byrs,frozen_meat_byrs,forzen_semi_byrs,wine_byrs,rice_byrs,egg_byrs,is_on_shelf,is_outstock,arranged_cnt,dt'.split(',')

# forecast_date = np.nan

def sku_to_train(df_train_raw, forecast_date):
    from datetime import timedelta
    b_1d_fc_dt = pd.to_datetime(forecast_date) - timedelta(1)
    b_30d_fc_dt = pd.to_datetime(forecast_date) - timedelta(30)
    # 过滤了很多SKU
    sku_30d_arr = df_train_raw[(df_train_raw['dt'] <= b_1d_fc_dt) & (df_train_raw['dt'] >= b_30d_fc_dt)]
    sku_30d_arr_sum = sku_30d_arr.groupby('sku_id', as_index=False)[['arranged_cnt']].sum()
    sku_train_list = sku_30d_arr_sum[sku_30d_arr_sum['arranged_cnt'] != 0]['sku_id'].tolist()
    print("ST02, sku_to_train, B: {}, A:{}".format(len(df_train_raw.sku_id.unique()), len(sku_train_list)))
    return sku_train_list


def preprocess_map(df_raw, forecast_date):
    df = df_raw.copy()
    nonan_data = []

    for sku_id, data in df.groupby('sku_id'):
        nonan_data.append(data[['tax_rate', 'csu_origin_price']].fillna(method='bfill').fillna(method='ffill'))
    df_nonan = pd.concat(nonan_data).reindex(index=df.index)
    df[['tax_rate', 'csu_origin_price']] = df_nonan

    df['festival_name'].fillna('无', inplace=True)
    df['western_festival_name'].fillna('无', inplace=True)

    df['seq_num'].fillna(0, inplace=True)
    df['redu_num'].fillna(0, inplace=True)
    df['pro_num'].fillna(0, inplace=True)

    mprice_nan = df[df['seq_price'].isnull() == True].index
    dprice_nan = df[df['discount_price'].isnull() == True].index
    df.loc[mprice_nan, 'seq_price'] = df.loc[mprice_nan, 'csu_origin_price']
    df.loc[dprice_nan, 'discount_price'] = df.loc[dprice_nan, 'csu_origin_price']

    df['price_diff_seq'] = df['csu_origin_price'] - df['seq_price']
    df['price_diff_discount'] = df['csu_origin_price'] - df['discount_price']
    df['price_diff_sd'] = df['discount_price'] - df['seq_price']

    #     df = df.sort_values('dt')

    df.loc[(df['avg_t'] > 100) | (df['avg_t'] < -100), 'avg_t'] = np.nan
    df.loc[(df['avg_t_b'] > 100) | (df['avg_t_b'] < -100), 'avg_t_b'] = np.nan

    avg_nonan = df[['avg_t', 'avg_t_b']].fillna(method='bfill')
    avg_nonan = avg_nonan.fillna(method='ffill')
    df[['avg_t', 'avg_t_b']] = avg_nonan

    df['weather'].fillna('晴', inplace=True)
    df['weather_b'].fillna('晴', inplace=True)

    byr_cols = [col for col in df.columns if '_byrs' in col]
    for col in byr_cols:
        df[col] = df[col].fillna(method='bfill')
        df[col] = df[col].fillna(method='ffill')
    return df


def his_sale_avg(raw_data, forecast_date):
    data = raw_data.copy().sort_values('dt')
    sku_data_list = []
    for sku, sku_data in data.groupby('sku_id'):
        df_sku = sku_data[sku_data.is_train == 1]
#         arr_list = df_sku.set_index('dt')['arranged_cnt'].asfreq('1D').ffill()
        arr_list = df_sku['arranged_cnt']
        roll_7d_avg = arr_list.rolling(7, min_periods=1).mean()
        roll_15d_avg = arr_list.rolling(15, min_periods=1).mean()
        roll_30d_avg = arr_list.rolling(30, min_periods=1).mean()        
        his_avg = (roll_7d_avg + roll_15d_avg + roll_30d_avg) / 3.
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
            arr_and_avg.append((avg_7d + avg_15d + avg_30d) / 3.)
            
        his_avg = np.concatenate([[his_avg.iloc[0]], his_avg, arr_and_avg[-19:]])
        if sku_data.shape[0] != len(his_avg):
            continue
        sku_data_list.append(pd.DataFrame({'sku_id': sku, 'dt': sku_data.dt, 'his_avg': his_avg}))
    data = data.merge(pd.concat(sku_data_list), how='left', on=['sku_id', 'dt'])
    return data


def predict_sp(df_test, sku_dic, drop_list, forecast_date, normal_feat_cols):
    forecast_sku_result = []
    pred_dates = df_test.dt.unique()
    dates_len = 20
    rst = []
    for sku in sku_dic:
        df_test_sku = df_test[df_test['sku_id'] == sku]
        X_test = df_test_sku.drop(drop_list, axis=1)
        rst.append(X_test[normal_feat_cols].values)
    return rst


def seasonal_count(df_total_raw, df_train_total, forecast_date):
    df_total = df_total_raw.copy()

    cat1_sale_mean = df_train_total.groupby('cat1_name', as_index=False).arranged_cnt.mean()
    cat2_sale_mean = df_train_total.groupby('cat2_name', as_index=False).arranged_cnt.mean()

    cat1_week_mean = df_train_total.groupby(['cat1_name', 'day_abbr'], as_index=False).arranged_cnt.mean()
    cat1_week_mean = cat1_week_mean.merge(cat1_sale_mean, how='left', on=['cat1_name'], suffixes=('_week', '_total'))
    cat1_week_mean['cat1_week_avg'] = cat1_week_mean['arranged_cnt_week'] / cat1_week_mean['arranged_cnt_total']
    df_total = df_total.merge(cat1_week_mean[['cat1_name', 'day_abbr', 'cat1_week_avg']], how='left', on=['cat1_name', 'day_abbr'])

    cat2_week_mean = df_train_total.groupby(['cat2_name', 'day_abbr'], as_index=False).arranged_cnt.mean()
    cat2_week_mean = cat2_week_mean.merge(cat2_sale_mean, how='left', on=['cat2_name'], suffixes=('_week', '_total'))
    cat2_week_mean['cat2_week_avg'] = cat2_week_mean['arranged_cnt_week'] / cat2_week_mean['arranged_cnt_total']
    df_total = df_total.merge(cat2_week_mean[['cat2_name', 'day_abbr', 'cat2_week_avg']], how='left', on=['cat2_name', 'day_abbr'])

    cat1_month_mean = df_train_total.groupby(['cat1_name', 'month'], as_index=False).arranged_cnt.mean()
    cat1_month_mean = cat1_month_mean.merge(cat1_sale_mean, how='left', on=['cat1_name'], suffixes=('_week', '_total'))
    cat1_month_mean['cat1_month_avg'] = cat1_month_mean['arranged_cnt_week'] / cat1_month_mean['arranged_cnt_total']
    df_total = df_total.merge(cat1_month_mean[['cat1_name', 'month', 'cat1_month_avg']], how='left', on=['cat1_name', 'month'])

    cat2_month_mean = df_train_total.groupby(['cat2_name', 'month'], as_index=False).arranged_cnt.mean()
    cat2_month_mean = cat2_month_mean.merge(cat2_sale_mean, how='left', on=['cat2_name'], suffixes=('_week', '_total'))
    cat2_month_mean['cat2_month_avg'] = cat2_month_mean['arranged_cnt_week'] / cat2_month_mean['arranged_cnt_total']
    df_total = df_total.merge(cat2_month_mean[['cat2_name', 'month', 'cat2_month_avg']], how='left', on=['cat2_name', 'month'])

    cat1_fevl_mean = df_train_total.groupby(['cat1_name', 'festival_name'], as_index=False).arranged_cnt.mean()
    cat1_fevl_mean.rename(columns={'arranged_cnt': 'cat1_fevl_avg'}, inplace=True)
    cat1_fevl_mean = cat1_fevl_mean.merge(cat1_sale_mean)
    cat1_fevl_mean['cat1_fevl_avg'] = cat1_fevl_mean.cat1_fevl_avg / cat1_fevl_mean.arranged_cnt
    del cat1_fevl_mean['arranged_cnt']
    df_total = df_total.merge(cat1_fevl_mean, how='left', on=['cat1_name', 'festival_name'])

    cat2_fevl_mean = df_train_total.groupby(['cat2_name', 'festival_name'], as_index=False).arranged_cnt.mean()
    cat2_fevl_mean.rename(columns={'arranged_cnt': 'cat2_fevl_avg'}, inplace=True)
    cat2_fevl_mean = cat2_fevl_mean.merge(cat2_sale_mean)
    cat2_fevl_mean['cat2_fevl_avg'] = cat2_fevl_mean.cat2_fevl_avg / cat2_fevl_mean.arranged_cnt
    del cat2_fevl_mean['arranged_cnt']
    df_total = df_total.merge(cat2_fevl_mean, how='left', on=['cat2_name', 'festival_name'])

    cat1_wfevl_mean = df_train_total.groupby(['cat1_name', 'western_festival_name'], as_index=False).arranged_cnt.mean()
    cat1_wfevl_mean.rename(columns={'arranged_cnt': 'cat1_wfevl_avg'}, inplace=True)
    cat1_wfevl_mean = cat1_wfevl_mean.merge(cat1_sale_mean)
    cat1_wfevl_mean['cat1_wfevl_avg'] = cat1_wfevl_mean.cat1_wfevl_avg / cat1_wfevl_mean.arranged_cnt
    del cat1_wfevl_mean['arranged_cnt']
    df_total = df_total.merge(cat1_wfevl_mean, how='left', on=['cat1_name', 'western_festival_name'])

    cat2_wfevl_mean = df_train_total.groupby(['cat2_name', 'western_festival_name'], as_index=False).arranged_cnt.mean()
    cat2_wfevl_mean.rename(columns={'arranged_cnt': 'cat2_wfevl_avg'}, inplace=True)
    cat2_wfevl_mean = cat2_wfevl_mean.merge(cat2_sale_mean)
    cat2_wfevl_mean['cat2_wfevl_avg'] = cat2_wfevl_mean.cat2_wfevl_avg / cat2_wfevl_mean.arranged_cnt
    del cat2_wfevl_mean['arranged_cnt']
    df_total = df_total.merge(cat2_wfevl_mean, how='left', on=['cat2_name', 'western_festival_name'])

    return df_total.fillna(0.)


def byrs_count(df_total_raw, df_train_total, col_list=None):

    df_total = df_total_raw.copy()

    for col in col_list:
        week_col_name = col + '_week_avg'
        month_col_name = col + '_month_avg'
        fevl_col_name = col + '_fevl_avg'
        wfevl_col_name = col + '_wfevl_avg'

        _mean = df_train_total.groupby('cat1_name', as_index=False)[[col]].mean()

        _week_mean = df_train_total.groupby(['cat1_name', 'day_abbr'], as_index=False)[[col]].mean()
        _week_mean = _week_mean.merge(_mean, how='left', on=['cat1_name'])
        _week_mean[week_col_name] = _week_mean.iloc[:, -2] / _week_mean.iloc[:, -1]

        _month_mean = df_train_total.groupby(['cat1_name', 'month'], as_index=False)[[col]].mean()
        _month_mean = _month_mean.merge(_mean, how='left', on=['cat1_name'])
        _month_mean[month_col_name] = _month_mean.iloc[:, -2] / _month_mean.iloc[:, -1]

        _fevl_mean = df_train_total.groupby(['cat1_name', 'festival_name'], as_index=False)[[col]].mean()
        _fevl_mean = _fevl_mean.merge(_mean, how='left', on=['cat1_name'])
        _fevl_mean[fevl_col_name] = _fevl_mean.iloc[:, -2] / _fevl_mean.iloc[:, -1]

        _wfevl_mean = df_train_total.groupby(['cat1_name', 'western_festival_name'], as_index=False)[[col]].mean()
        _wfevl_mean = _wfevl_mean.merge(_mean, how='left', on=['cat1_name'])
        _wfevl_mean[wfevl_col_name] = _wfevl_mean.iloc[:, -2] / _wfevl_mean.iloc[:, -1]

        df_total = df_total.merge(_week_mean[['cat1_name', 'day_abbr', week_col_name]], how='left', on=['cat1_name', 'day_abbr'])
        df_total = df_total.merge(_month_mean[['cat1_name', 'month', month_col_name]], how='left', on=['cat1_name', 'month'])
        df_total = df_total.merge(_fevl_mean[['cat1_name', 'festival_name', fevl_col_name]], how='left', on=['cat1_name', 'festival_name'])
        df_total = df_total.merge(_wfevl_mean[['cat1_name', 'western_festival_name', wfevl_col_name]], how='left', on=['cat1_name', 'western_festival_name'])
    return df_total.fillna(0.).drop(col_list, axis=1)

def cal_mean_old(df_train_total, forecast_date):
    from datetime import timedelta
    start_dt = pd.to_datetime(forecast_date) - timedelta(7)
    end_dt = pd.to_datetime(forecast_date) - timedelta(1)
    df_sku_cnt = df_train_total[(df_train_total.dt >= start_dt) & (df_train_total.dt <= end_dt)]
    arr_mean = df_sku_cnt.groupby(['bu_id', 'wh_id', 'cat1_name', 'sku_id'], as_index=False)['arranged_cnt'].sum()
    group_size = df_sku_cnt.groupby('sku_id', as_index=False).size().reindex(index=arr_mean.sku_id)
    arr_mean['arr_mean'] = arr_mean['arranged_cnt'] / group_size.values
    return arr_mean

def drop_Outliers(train_raw, forecast_date):
    train_data = train_raw.copy()
    droped_index = []
    for sku in train_data.sku_id.unique():
        df_sku = train_data[train_data.sku_id == sku]
        if len(df_sku) < 7:
            continue
        data_normal = df_sku[(df_sku['pro_num'] == 0) & (df_sku['seq_num'] == 0) & (df_sku['redu_num'] == 0)]['arranged_cnt']
        data_pro = df_sku[(df_sku['pro_num'] != 0) | (df_sku['seq_num'] != 0) | (df_sku['redu_num'] != 0)]['arranged_cnt']

        if len(data_normal) >= 7:
            for i in range(7, len(data_normal) + 1):
                window = data_normal.iloc[i - 7:i]
                avg = window.mean()
                droped_index.extend(window[window > avg * 2.].index)
        if len(data_pro) >= 7:
            for j in range(7, len(data_pro) + 1):
                window = data_pro.iloc[j - 7:j]
                avg = window.mean()
                droped_index.extend(window[window > avg * 2.].index)
    train_raw.loc[train_raw.index.isin(droped_index), 'is_outliers'] = 1
    return train_raw

def features_create(data, train_data, dt_field=None, dum_field_list=None, map_field_list=None, forecast_date=None):
    if dt_field:
        data['year'] = pd.to_datetime(data[dt_field]).dt.year
        data['month'] = pd.to_datetime(data[dt_field]).dt.month
        data['day'] = pd.to_datetime(data[dt_field]).dt.day
        train_data['month'] = pd.to_datetime(train_data[dt_field]).dt.month
    data = his_sale_avg(data, forecast_date)
    data = seasonal_count(data, train_data, forecast_date)
    byr_cols = [col for col in data.columns if '_byrs' in col]
    data = byrs_count(data, train_data, col_list=byr_cols)
    weather2int = {'雨':0,'NULL':5,'霾':7,'中雨':2,'雪':3,'雾':6,'大雨':4,'小雨':1,'晴':5,'小雨':1,'大雪':3,'NULL':5,'中雪':3,'暴雨':4,'晴':5,'扬沙':5,'多云':5,'阵雨':2,'雷阵雨':2,'大雨':4,'小雪':3,'雨夹雪':3,'大暴雨':4,'阴':0,'雾':6,'阵雪':3,'浮尘':5,'中雨':2,'霾':7}

    data['weather'] = data['weather'].map(weather2int)
    data['weather_b'] = data['weather_b'].map(weather2int)

    data[map_field_list] = data[map_field_list].astype('category')
    cat_columns = data.select_dtypes(['category']).columns
    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
    return data[data.is_train == 1], data[data.is_train == 0]

def sku_to_forecast(df_pred_raw):
    cat_list = df_pred_raw['cat1_name'].unique()
    cat_sku_map = {}
    day_cat_sku_temp = df_pred_raw[df_pred_raw['dt'] == forecast_date]
    sku_dic = {}
    for cat in cat_list:
        cat_sku_map[cat] = []
        df_temp = day_cat_sku_temp[day_cat_sku_temp['cat1_name'] == cat]
        sku_temp_list = df_temp['sku_id'].unique().tolist()
        cat_sku_map[cat].extend(sku_temp_list)
        sku_temp_dic = {sku: cat for sku in sku_temp_list}
        sku_dic.update(sku_temp_dic)
    return sku_dic

def train_map(line, input_forecast_date, schema_cols, normal_feat_cols):
    global forecast_date
    forecast_date = input_forecast_date
    wh_id = int(line[0][1])
    total_df = pd.DataFrame(line[1], columns=schema_cols)    
    total_df['dt'] = pd.to_datetime(total_df['dt'])
    total_df['date'] = total_df['dt'].apply(lambda x: x.strftime('%Y%m%d'))
    total_df.sort_values('dt', inplace=True)
    total_df['is_outliers'] = 0
    df_train_raw = total_df[total_df.is_train == 1]
    df_pred_raw = total_df[total_df.is_train == 0]
    if df_train_raw.shape[0] == 0:
        return []
    sku_size = df_pred_raw.groupby('sku_id').size()
    less_20d_skus = sku_size[sku_size < 20].index
    df_train_raw = df_train_raw[~df_train_raw.sku_id.isin(less_20d_skus)]
    df_pred_raw = df_pred_raw[~df_pred_raw.sku_id.isin(less_20d_skus)]
    if (df_train_raw.shape[0] == 0) | (df_pred_raw.shape[0] == 0):
        return []
    df_train_total = preprocess_map(df_train_raw, forecast_date)
    trained_sku = sku_to_train(df_train_raw, forecast_date)
    df_train = df_train_total[df_train_total['sku_id'].isin(trained_sku)]
    if df_pred_raw[df_pred_raw['sku_id'].isin(trained_sku)].shape[0] == 0:
        print('[ST02] skip \n{}'.format(df_pred_raw['sku_id,wh_id,is_train,dt'.split(',')].head(1).T))
        return []
    try:
        df_test = preprocess_map(df_pred_raw[df_pred_raw['sku_id'].isin(trained_sku)], forecast_date)
        df_train = drop_Outliers(df_train, forecast_date)
        df_test = df_test[df_test.sku_id.isin(df_train.sku_id.unique())]
        sku_dic = sku_to_forecast(df_pred_raw)
        df_total_raw = pd.concat([df_train, df_test])
        dum_list = None
        map_list = 'day_abbr,festival_name,western_festival_name'.split(',')
        df_train, df_test = features_create(data=df_total_raw, train_data=df_train_total, dt_field='dt', dum_field_list=dum_list, map_field_list=map_list, forecast_date=forecast_date)
        if df_train is None:
            return []
        drop_list = ['cat1_name', 'cat2_name', 'date', 'brand_name']
        train_feat = df_train.drop(drop_list, axis=1)[normal_feat_cols].values
        predict_feat = predict_sp(df_test, sku_dic, drop_list, forecast_date, normal_feat_cols)
        return [train_feat, predict_feat]
    except Exception as e:
        print("[ST02] E {}\n{}".format(e, df_test['sku_id,wh_id,is_train,dt'.split(',')].head(1).T))
        traceback.print_exc()
        return []