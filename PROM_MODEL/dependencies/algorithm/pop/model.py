from __future__ import unicode_literals

# 设置随机性
import numpy as np
import random
import os
import tensorflow as tf
seed=44
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = 'true'
np.random.seed(seed)
tf.random.set_seed(seed)

import dependencies.algorithm.pop.preprocess as fu
from dependencies.algorithm.pop.filterOutliers import quantile_filter
from dependencies.algorithm.pop.BHT_ARIMA import BHTARIMA
from dependencies.algorithm.pop.BHT_ARIMA.util.utility import get_index
import numpy as np
import pandas as pd


###########################################wma case#23  utils###########################################################
def robust_op(ll, op=np.nanmean, default=0.0):
    r = op(ll)
    return (default if np.isnan(r) else r)
def rand_wma(_rdd, configs):
    """
    使用随机抽样的平均值作为预测值
    :param _rdd
    :param configs
    :return:
    """
    import pandas as pd
    import numpy as np
    import random
    sku_data = pd.DataFrame(_rdd[1], columns=configs['cnt_data_columns']).sort_values('dt')
    sku_data = sku_data[((sku_data.is_on_shelf == 1) | (sku_data.arranged_cnt > 0))]

    forecast_span = configs['pred_len']
    checking_len = configs['checking_len']
    sales_list = sku_data[configs['label']]
    unique_id = _rdd[0]

    # 最大值
    sales_upper_bound = np.percentile(np.unique(sales_list), 100)
    # 最近的历史天数
    sales_clean = sales_list[sales_list <= sales_upper_bound][-checking_len:]
    # results = [0.0] * forecast_span
    # 不做 0 值预测？？？
    results = [random.uniform(0.001, 0.01) for i in range(forecast_span)]
    # 用0补全sales_clean到14天
    records_low_bound = 7
    if len(sales_clean) < records_low_bound:
        sales_clean = np.append([0.0] * (records_low_bound - len(sales_clean)), sales_clean)
    for i in range(1, forecast_span + 1):
        total_weight = 0.0
        weighted_sale = 0.0
        # 重sale_clean 中抽样40%
        sales_sample = []
        for sale in sales_clean:
            if random.random() < 0.4:
                sales_sample.append(sale)
        # weight 越近越大 # 稍后尝试
        for j, sale in enumerate(sales_sample):
            weight = j
            total_weight += weight
            weighted_sale += float(sale * weight)
        sale = (0.0 if total_weight == 0.0 else weighted_sale / total_weight)
        sales_clean = np.append(sales_clean,[sale])
        # 随机抽样的平均值
        # sale = robust_op(sales_sample)
        results[(i - 1)] += sale

    unique_ids = unique_id.split('_')
    router_data = pd.DataFrame(
        list(zip([unique_ids[0]] * forecast_span, [unique_ids[1]] * forecast_span, [unique_ids[2]] * forecast_span,
                 configs['pred_dates'].strftime('%Y%m%d'), results)),
        columns=['bu_id', 'wh_id', 'sku_id', 'fc_dt', 'predication'])

    return router_data['bu_id,wh_id,sku_id,fc_dt,predication'.split(',')].values
def wma(_rdd, configs):
    """
    使用随机抽样的平均值作为预测值
    :param _rdd
    :param configs
    :return:
    """
    import pandas as pd
    import numpy as np
    import random
    from dependencies.algorithm.pop.filterOutliers import quantile_filter

    sku_data = pd.DataFrame(_rdd[1], columns=configs['cnt_data_columns']).sort_values('dt')
    sku_data = sku_data[((sku_data.is_on_shelf == 1) | (sku_data.arranged_cnt > 0))]

    forecast_span = configs['pred_len']
    checking_len = configs['checking_len']
    sales_list = sku_data[configs['label']]
    unique_id = _rdd[0]

    # 最大值
    sales_upper_bound = np.percentile(np.unique(sales_list), 100)
    # 最近的历史天数
    sales_clean = sales_list[sales_list <= sales_upper_bound][-checking_len:]
    # 不做 0 值预测
    results = [random.uniform(0.001, 0.01) for i in range(forecast_span)]
    sales_clean = quantile_filter(sales_clean)
    if len(sales_clean)<7:
        results = [np.mean(sales_clean)]*forecast_span
    else:
        results = [np.mean(sales_clean[-4:])]*forecast_span

    results = [np.max([0, i]) for i in results]  # 过滤nan 和 负数
    #construct final output
    unique_ids = unique_id.split('_')
    router_data = pd.DataFrame(
        list(zip([unique_ids[0]] * forecast_span, [unique_ids[1]] * forecast_span, [unique_ids[2]] * forecast_span,
                 configs['pred_dates'].strftime('%Y%m%d'), results)),
        columns=['bu_id', 'wh_id', 'sku_id', 'fc_dt', 'predication'])

    return router_data['bu_id,wh_id,sku_id,fc_dt,predication'.split(',')].values
###########################################wma case#23  utils###########################################################
###########################################ses_longtail_baseline utils###########################################################
def ses_new(_rdd, configs):
    """
    使用随机抽样的平均值作为预测值
    :param _rdd
    :param configs
    :return:
    """
    import pandas as pd
    import numpy as np
    import random
    from copy import copy
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    from dependencies.algorithm.pop.filterOutliers import quantile_filter

    forecast_span = configs['pred_len']
    unique_id = _rdd[0]

    sku_data = pd.DataFrame(_rdd[1], columns=configs['cnt_data_columns']).sort_values('dt')
    origin_saled_list = copy(sku_data[configs['label']].values)  # saved for optimise T0
    print('origin_saled_list', origin_saled_list)
    # 首先第一步过滤一些极其异常的sku
    sku_data[configs['label']] = quantile_filter(sku_data[configs['label']], q=0.99,
                                                 backfill=0.9).values  # 数据量少，纸过滤极其异常的case

    if sku_data.shape[0] <= 2:  #
        print('sku_data.shape[0]<=2')
        results = [sku_data[configs['label']].mean()] * forecast_span
    else:
        # 然后处理掉，一直没有销售的商品，用最近的均值，等第一天有了后面就有了
        sku_data = sku_data.assign(same_count=sku_data.groupby(
            sku_data[configs['label']].ne(sku_data[configs['label']].shift()).cumsum()).cumcount())
        if (sku_data.iloc[[-1]][configs['label']].values[0] == 0) & (
                sku_data.iloc[[-1]].same_count.values[0] >= 4):  # 最近连续4天不上
            print('最近连续4天不上')
            results = [sku_data.tail(3)[configs['label']].mean()] * forecast_span
        else:
            # 然后使用可售的商品做预测
            sku_data = sku_data[((sku_data.is_on_shelf == 1) | (sku_data.arranged_cnt > 0))]
            sales_list = sku_data[configs['label']].values

            # 最大值
            sales_upper_bound = np.percentile(np.unique(sales_list), 100)
            # 最近的历史天数
            sales_clean = sales_list[sales_list <= sales_upper_bound][-31:]
            print('然后使用可售的商品做预测', sales_clean)
            try:
                results = SimpleExpSmoothing(sales_clean).fit(smoothing_level=0.71).forecast()[:forecast_span]
            except:
                # 异常时使用过去n天平均
                print('异常时使用过去n天平均')
                results = [sales_clean[sales_clean > 0][-3:].mean()] * forecast_span
            # opimise t0
            # optimisie t0
            if (len(origin_saled_list) <= 15):  # 刚买了很短,昨天忽然很高的销售，促销不会只有一天；只为了优化T0
                last_day = origin_saled_list[-1]
                last_day_2 = origin_saled_list[-2]
                if (last_day_2 >= 1) & ((last_day // last_day_2) >= 5):  # first day saled!!# saled up!!
                    print('优化T0', last_day, last_day_2)
                    results[0] = np.mean([results[0], last_day])
    results = pd.Series(results).fillna(0).tolist()  # filter T0
    results = [np.max([random.uniform(0.001, 0.01), i]) for i in results]  # 过滤nan 和 负数

    unique_ids = unique_id.split('_')
    return_data = pd.DataFrame(
        list(zip([unique_ids[0]] * forecast_span, [unique_ids[1]] * forecast_span, [unique_ids[2]] * forecast_span,
                 configs['pred_dates'].strftime('%Y%m%d'), results)),
        columns=['bu_id', 'wh_id', 'sku_id', 'fc_dt', 'predication'])

    return return_data['bu_id,wh_id,sku_id,fc_dt,predication'.split(',')].values
def ses(_rdd, configs):
    """
    :param _rdd
    :param configs
    :return:
    """
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    import random
    sku_data = pd.DataFrame(_rdd[1], columns=configs['cnt_data_columns']).sort_values('dt')
    # filtering able sell days , then smooth weighted
    sku_data = sku_data[((sku_data.is_on_shelf == 1) | (sku_data.arranged_cnt > 0))]

    forecast_span = configs['pred_len']
    sales_list = sku_data[configs['label']]
    unique_id = _rdd[0]

    # 最大值
    sales_upper_bound = np.percentile(np.unique(sales_list), 100)
    # 最近的历史天数
    sales_clean = sales_list[sales_list <= sales_upper_bound][-31:]

    try:
        results = SimpleExpSmoothing(sales_clean).fit(smoothing_level=0.4).forecast()[:forecast_span]
    except:
        # 异常时使用过去n天平均
        results = [sales_clean[-3:].mean()] * forecast_span

    results = [np.max([0, i]) for i in results]  # 过滤nan 和 负数
    unique_ids = unique_id.split('_')
    return_data = pd.DataFrame(
        list(zip([unique_ids[0]] * forecast_span, [unique_ids[1]] * forecast_span, [unique_ids[2]] * forecast_span,
                 configs['pred_dates'].strftime('%Y%m%d'), results)),
        columns=['bu_id', 'wh_id', 'sku_id', 'fc_dt', 'predication'])

    return return_data['bu_id,wh_id,sku_id,fc_dt,predication'.split(',')].values
###########################################ses_longtail_baseline utils###########################################################

###########################################longtail optimize bht_arima###########################################################

###########################################longtail optimize bht_arima###########################################################
def bht_arima(_rdd, configs):
    """
    使用bht_arima预测
    :param _rdd
    :param configs
    :return:
    """
    import pandas as pd
    import numpy as np
    import datetime
    from dependencies.algorithm.pop.filterOutliers import quantile_filter
    from dependencies.algorithm.pop.BHT_ARIMA import BHTARIMA
    from dependencies.algorithm.pop.BHT_ARIMA.util.utility import get_index
    import random

    group_data = pd.DataFrame(_rdd[1], columns=configs['bht_arima_columns']).sort_values('dt')
    forecast_span = configs['pred_len']
    checking_len = configs['checking_len']
    fitting_len = configs['fitting_len']
    group_label = _rdd[0]
    seed = configs['seed']

    ## 这个传入应该就已经卡掉了，所以是没有影响的，在这里保留也可以
    group_data = group_data[(group_data[configs['label']] > 0) | (group_data.is_on_shelf == 1)][
        ['unique_id', 'dt', configs['label']]]

    print('start trainging hahhah')
    # predict span
    final_pred = []
    for i in range(forecast_span):
        # construct matrix for fitting'
        print('construct matrix for fitting with index is --',i,forecast_span,(configs['fc_dt'] + datetime.timedelta(days=i)).strftime('%Y%m%d'))
        sequence_len = fitting_len
        keys = []
        sequences = []
        for k, g in group_data.groupby(['unique_id']):
            keys.append(k)
            k_g_sequence = g[configs['label']].to_list()[-sequence_len:]
            k_g_sequence.append(k_g_sequence[-1])  # adding last for real value # adding pred？
            # filter outliers
            k_g_sequence = list(quantile_filter(k_g_sequence))
            # backfill for matrix decom
            if len(k_g_sequence) <= (sequence_len + 1):
                for j in range(sequence_len + 1 - len(k_g_sequence)):
                    # k_g_sequence.insert(0, k_g_sequence[0])
                    k_g_sequence.insert(0, k_g_sequence[-1]) # 过拟合昨天的
            sequences.append(k_g_sequence)

        keys_df = pd.DataFrame(keys, columns=['unique_id'])
        # prepare data
        ori_ts = np.array(sequences)
        print("shape of data: {}".format(ori_ts.shape))
        print("This dataset have {} series, and each serie have {} time step".format(ori_ts.shape[0], ori_ts.shape[1]))

        # parameters setting fixed for now
        taus = [ori_ts.shape[0], 5]  # MDT-rank
        Rs = [5, 5]  # tucker decomposition ranks
        ts = ori_ts[:, :-1]  # training data,
        k = 300  # iterations
        tol = 0.001  # stop criterion
        Us_mode = 4  # orthogonality mode
        label = ori_ts[:, -1]  # label, take the last time step as label
        p, d, q = 3, 1, 1  # search_params(ts)

        # fit model program
        model = BHTARIMA(ts[:, 2:], p, d, q, taus, Rs, k, tol,seed=seed, verbose=0, Us_mode=Us_mode)
        result, _ = model.run()
        pred = result[:, -1]
        print("Evaluation index: \n{}".format(get_index(pred, label)))
        keys_df['fc_dt'] = (configs['fc_dt'] + datetime.timedelta(days=i)).strftime('%Y%m%d')
        keys_df['dt'] = (configs['fc_dt'] + datetime.timedelta(days=i)).strftime('%Y%m%d')
        keys_df['predication'] = (result[:, -1] + result[:, -2] + result[:, -3]+ result[:, -7]) / 4
        keys_df['predication'] = keys_df['predication'].apply(lambda x: random.uniform(0.001, 0.01) if x <= 0 else x)
        keys_df[configs['label']] = keys_df['predication']
        keys_df['bu_id'] = keys_df.unique_id.apply(lambda x: x.split('_')[0])
        keys_df['wh_id'] = keys_df.unique_id.apply(lambda x: x.split('_')[1])
        keys_df['sku_id'] = keys_df.unique_id.apply(lambda x: x.split('_')[2])
        final_pred.append(keys_df)
        group_data = pd.concat([group_data, keys_df[['unique_id', 'dt', configs['label']]]])  # 拼接然后做T1 etc的预测

    # agg all predictions
    final_pred_df = pd.concat(final_pred)
    final_pred_df['predication'] = final_pred_df['predication'].apply(lambda x:random.uniform(0.001, 0.01) if x<=0 else x)
    return final_pred_df['bu_id,wh_id,sku_id,fc_dt,predication'.split(',')].values.tolist()

###########################################longtail optimize bht_arima###########################################################
###########################################arima case#23  utils###########################################################
def arima_one_model(_rdd, configs):
    import itertools
    import pandas as pd
    import numpy as np
    from dependencies.algorithm.pop.filterOutliers import quantile_filter
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # prepare data
    sku_data = pd.DataFrame(_rdd[1], columns=configs['cnt_data_columns']).sort_values('dt')
    sku_data['ds'] = pd.to_datetime(sku_data['dt'].astype(str))
    sku_data['y'] = sku_data[configs['label']]
    sku_data = sku_data[((sku_data.arranged_cnt > 0) | (sku_data.is_on_shelf == 1))]
    forecast_span = configs['pred_len']
    unique_id = _rdd[0]

    #     print('fitting',sku_data)
    try:
        # 总共有销售的只有3次，直接均值
        if sku_data['ds'].count() <= 3:
            if sku_data['ds'].count() <= 0:  # 没有交易过的商品
                output = [0]
            else:
                output = [sku_data['y'].mean()]
        else:
            p = d = q = range(0, 2)
            pdq = list(itertools.product(p, d, q))
            seasonal_pdq = [(x[0], x[1], x[2], 3) for x in list(itertools.product(p, d, q))]
            df_group = sku_data[['ds', 'y']].reset_index(drop=True)
            df_group.set_index(pd.to_datetime(df_group['ds']), inplace=True)
            df_group['y'] = df_group['y'].astype(float)
            df_group['y'] = quantile_filter(df_group['y'])
            AIC = []
            parm_ = []
            parm_s = []
            for param in pdq:
                # for param_seasonal in seasonal_pdq:
                try:
                    mod = SARIMAX(df_group['y'],
                                  order=param,
                                  #seasonal_order=param_seasonal, # default no seasonal
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
                    results = mod.fit()
                    AIC.append(results.aic)
                    parm_.append(param)
                    # parm_s.append(param_seasonal)

                #                         print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                except:
                    continue
        #             print('AIC is ',AIC)
            pos = AIC.index(np.min(AIC))
            #             print('AIC is ',pos,str(AIC))
            #             print('uniqueid is ',unique_id,df_group['y'])
            stepwise_fit = SARIMAX(df_group['y'],
                                   order=parm_[pos],
                                   # seasonal_order=parm_s[pos],
                                   enforce_stationarity=False,
                                   enforce_invertibility=False)

            output = stepwise_fit.fit().get_forecast(forecast_span).predicted_mean.reset_index()[0].values
            del stepwise_fit
    except:
        output = [sku_data['y'].mean()]

    #     print('output is ',output,sku_data)
    output = [np.max([0, i]) for i in output]  # 过滤nan 和 负数

    # construct output
    unique_ids = unique_id.split('_')
    return_data = pd.DataFrame(
        list(zip([unique_ids[0]] * forecast_span, [unique_ids[1]] * forecast_span, [unique_ids[2]] * forecast_span,
                 configs['pred_dates'].strftime('%Y%m%d'), output)),
        columns=['bu_id', 'wh_id', 'sku_id', 'fc_dt', 'predication'])
    #     print('output is',output,return_data)

    return return_data['bu_id,wh_id,sku_id,fc_dt,predication'.split(',')].values

###########################################arima case#23  utils###########################################################

###########################################xbg case#4  utils###########################################################

date_shift = 2
shift_cols = []

for i in range(1, date_shift * 7 + 1, 7):
    # for i in range(1, date_shift + 1):
    shift_cols += "rolling_mean_1_shift_{i},rolling_mean_3_shift_{i},rolling_mean_7_shift_{i},rolling_std_7_shift_{i}".format(
        i=i).split(',')

raw_id_features = "wh_id,bu_id,sku_id,cat1_id,cat2_id,brand_id".split(',')
raw_price_features = "origin_price,seq_num,seq_price,csu_redu_num,csu_redu_mdct,csu_redu_adct,cir_redu_num,cir_redu_mdct,cir_redu_adct".split(
    ',')
raw_date_features = "is_work_day,is_weekend,is_holiday".split(',')
raw_byrs_features = "new_byrs,old_byrs,poultry_byrs,vege_fruit_byrs,egg_aquatic_byrs,standard_byrs,vege_byrs,meet_byrs,frozen_meat_byrs,forzen_semi_byrs,wine_byrs,rice_byrs,egg_byrs".split(
    ',')
raw_stock_features = "is_outstock".split(',')

raw_features = raw_id_features + raw_price_features + raw_date_features + raw_byrs_features + raw_stock_features

feature_cols = raw_features + shift_cols


def reindex_date(tto):
    import copy
    import pandas as pd
    import numpy as np
    tt = copy.deepcopy(tto)
    os = tt.arranged_cnt.sum()
    oss = tt.shape[0]
    tt['dt_dt'] = pd.to_datetime(tt['dt'].astype(str))
    tt.sort_values('dt_dt', inplace=True)
    dt_dt_list = tt.dt_dt.values
    tt = tt.set_index('dt_dt')

    for i in range(1, date_shift * 7 + 1, 7):
        tt['shift_arranged_cnt_{}'.format(i)] = tt['arranged_cnt'].shift(i)
        tt['rolling_mean_1_shift_{}'.format(i)] = tt['shift_arranged_cnt_{}'.format(i)].rolling(1).mean()
        tt['rolling_mean_3_shift_{}'.format(i)] = tt['shift_arranged_cnt_{}'.format(i)].rolling(3).mean()
        tt['rolling_mean_7_shift_{}'.format(i)] = tt['shift_arranged_cnt_{}'.format(i)].rolling(7).mean()
        tt['rolling_mean_14_shift_{}'.format(i)] = tt['shift_arranged_cnt_{}'.format(i)].rolling(14).mean()
        tt['rolling_mean_28_shift_{}'.format(i)] = tt['shift_arranged_cnt_{}'.format(i)].rolling(28).mean()
        tt['rolling_std_7_shift_{}'.format(i)] = tt['shift_arranged_cnt_{}'.format(i)].rolling(7).std()
        tt['rolling_std_14_shift_{}'.format(i)] = tt['shift_arranged_cnt_{}'.format(i)].rolling(14).std()
        tt['rolling_std_28_shift_{}'.format(i)] = tt['shift_arranged_cnt_{}'.format(i)].rolling(28).std()
        tt.drop('shift_arranged_cnt_{}'.format(i), inplace=True, axis=1)
    mps_up = 3 * tt['arranged_cnt'].rolling(window=14).mean()
    tt['is_outliers'] = 0
    tt.loc[(tt['arranged_cnt'] > mps_up) & (tt['arranged_cnt'] > 10), 'is_outliers'] = 1
    tt.loc[tt.is_outliers == 1, 'arranged_cnt'] = 3 * tt['arranged_cnt'].rolling(window=14, center=True).mean()
    rst = tt[tt.index.isin(dt_dt_list)]
    return rst


def compute_mapd(predicts, reals):
    import numpy as np, pandas as pd
    reals = reals.get_label()
    ape_list = np.where(reals == 0.0, np.where(predicts < 1, 0, 1), np.abs(reals - predicts))
    total_reals = np.nansum(reals)
    total_abs_diff = np.nansum(np.abs(reals - predicts))
    wmape = np.sum(ape_list * reals) / total_reals if total_reals != 0.0 else total_abs_diff
    return 'wmape', wmape


def asy_mse(y_true, y_pred):
    import numpy as np, pandas as pd
    theta = 0.5
    grad = np.where(y_pred <= y_true + 1, y_pred - y_true, theta * (y_pred - y_true))
    hess = np.where(y_pred <= y_true + 1, 1, theta)
    return grad, hess


def model_train(df_train, forecast_date, **kwargs):
    import numpy as np, pandas as pd
    from xgboost.sklearn import XGBRegressor
    print('inside model_train ', df_train.shape)
    #     df_train.dropna(inplace=True, subset=feature_cols + ['arranged_cnt']) # 有null的就不训练了。。
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_train = df_train.fillna(df_train.mean())
    df_train.drop('dt_dt,sku_id,wh_id,bu_id'.split(','), axis=1, inplace=True)
    df_train.reset_index(inplace=True)
    val_list = []

    print('inside model_train ', df_train.shape)

    for wh_id in df_train.wh_id.unique():
        wh_id_train = df_train[df_train.wh_id == wh_id]
        for cnt_band in wh_id_train.brand_id.unique():
            cnt_band_train = wh_id_train[wh_id_train.brand_id == cnt_band]
            sample_num = max(1, int(len(cnt_band_train) * 0.05))
            val_list.append(cnt_band_train.sample(sample_num))

    val_data = pd.concat(val_list)
    val_data = val_data.sort_values('dt_dt').groupby('bu_id,wh_id,sku_id'.split(',')).apply(
        lambda x: x.tail(7)).reset_index(level='bu_id,wh_id,sku_id'.split(','), drop=True)
    train_data = df_train.drop(index=val_data.index, axis=0)
    if kwargs['debug']:
        train_data[feature_cols + ['arranged_cnt']].to_csv('train_data.csv', index=False)
        val_data[feature_cols + ['arranged_cnt']].to_csv('val_data.csv', index=False)
    X_train = train_data[feature_cols].astype(float)
    y_train = train_data['arranged_cnt'].astype(float)
    X_val = val_data[feature_cols].astype(float)
    y_val = val_data['arranged_cnt'].astype(float)
    print('===x_train.shape_0:{}==='.format(X_train.shape[0]))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_train.head(1).T)

    parms = {'objective': asy_mse, 'learning_rate': 0.05, 'n_jobs': 12, 'nthread': 12, 'n_estimators': 800,
             'max_depth': 12, 'subsample': 0.85, 'colsample_bytree': 0.9, 'seed': 42, 'min_child_weight': 0,
             'disable_default_eval_metric': 1}
    bst = XGBRegressor(**parms).fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=True,
                                    eval_metric=compute_mapd)
    if bst.best_ntree_limit < 40:
        bst = XGBRegressor(**parms).fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20,
                                        verbose=True, eval_metric='rmse')
        print("eval_metric=rmse")

    if kwargs['debug']:
        fi_dict = {}
        tmp_fi = bst.feature_importances_
        for i in range(len(tmp_fi)):
            fi_dict[feature_cols[i]] = tmp_fi[i]
        tmp_df = pd.Series(fi_dict)
        tmp_df = tmp_df.round(4)
        tmp_df.sort_values(ascending=False).to_csv('feature_importance.csv', header=None)

    return bst


def predict(df_test, model):
    import pandas as pd
    import numpy as np
    forecast_sku_result = []
    dates_len = 20
    X_test = df_test[feature_cols].astype(float)
    if len(X_test) == 0:
        forecast_result = [0.] * dates_len
    else:
        forecast_result = model.predict(X_test, ntree_limit=model.best_ntree_limit)
    return forecast_result


def train_or_forecast(df_train, df_predict_raw, forecast_date, **kwargs):
    import pandas as pd
    import numpy as np
    import time, sys
    import random
    from datetime import datetime

    if df_train.shape[0] == 0:
        return [None]
    df_train['dt_dt'] = pd.to_datetime(df_train['dt'])
    if kwargs['is_train']:
        model = model_train(df_train, forecast_date, debug=kwargs['debug'])
        return model
    else:
        # read model, predict and intercept
        # 取训练集中每个SKU仓记录的最后一天真实记录
        tmp_df = df_train.drop('sku_id,wh_id,bu_id,dt_dt'.split(','), axis=1).reset_index().sort_values(
            'dt_dt').groupby('sku_id,wh_id,bu_id'.split(',')).tail(1)
        jj = df_predict_raw.set_index('sku_id,wh_id,bu_id'.split(',')).join(
            tmp_df.set_index('sku_id,wh_id,bu_id'.split(',')), rsuffix='_tmp', how='left').reset_index()
        jj = jj[jj['dt_dt'] == fu.get_date(forecast_date, -1, '%Y%m%d', '%Y-%m-%d')]
        df_predict = jj.dropna(subset=['dt_dt'])
        if df_predict.shape[0] == 0:
            return [None]
        df_predict['fcst_rst'] = predict(df_predict, kwargs['model'])

        arranged_cnt_df = df_train['arranged_cnt'].reset_index()
        avg_df = arranged_cnt_df.sort_values("dt_dt").groupby("sku_id,wh_id,bu_id".split(',')).apply(
            lambda df: df['arranged_cnt'].tail(14).mean()).reset_index().rename({0: 'new_his_avg'}, axis=1)
        avg_df.set_index('sku_id,wh_id,bu_id'.split(',')).join(df_predict.set_index('sku_id,wh_id,bu_id'.split(',')))
        df_predict = pd.merge(avg_df, df_predict, how='right', on='sku_id,wh_id,bu_id'.split(','))

        df_predict['his_avg'] = df_predict['new_his_avg'].astype(float)
        df_predict['inter_fcst_rst'] = df_predict['fcst_rst']
        jj = df_predict['sku_id,dt,wh_id,bu_id,inter_fcst_rst,fcst_rst,his_avg,cat1_id'.split(',')]
        if kwargs['debug']:
            df_predict[feature_cols + 'inter_fcst_rst,fcst_rst,dt'.split(',')].to_csv('predict_data.csv', index=False)
        jj.loc[jj['inter_fcst_rst'] > 3 * jj['his_avg'], 'inter_fcst_rst'] = 3 * jj['his_avg']
        jj.loc[jj['inter_fcst_rst'] < 1 / 3. * jj['his_avg'], 'inter_fcst_rst'] = jj['his_avg'] / 3.
        return jj['bu_id,wh_id,sku_id,dt,inter_fcst_rst'.split(',')].values


# Spark入口函数
def format_line(line, schema_cols, forecast_date, is_train=True, debug=False):
    import pandas as pd
    import numpy as np
    import base64
    import zlib as zl
    import _pickle as cp
    import pickle

    if is_train:
        cat1_id_str = '_'.join(map(str, [line[0]]))
        model_keyword = 'DA-WH_CAT1_{}'.format(cat1_id_str)
        swdf = pd.DataFrame(list(line[1]), columns=schema_cols)
        swdf = swdf[['dt', 'arranged_cnt', 'is_train'] + raw_features].apply(pd.to_numeric,
                                                                             errors='ignore')  # filtering only using features
        print('start training log', model_keyword, swdf.shape)

        if type(forecast_date) != int:
            forecast_date_int = int(forecast_date.strftime('%Y%m%d'))

        tdf = swdf[(swdf.is_train == 1) & (swdf.dt < forecast_date_int)].reset_index()  # 过滤掉今天的真实值，如果回测的话
        tdf.drop_duplicates(subset='sku_id,wh_id,bu_id,dt'.split(','), inplace=True)
        shift_tdf = tdf.groupby('sku_id,wh_id,bu_id'.split(",")).apply(reindex_date)  # 用时很长

        if debug:
            with open('objs_t.pkl', 'wb') as f: pickle.dump([shift_tdf, pdf], f)
        tdf = pd.DataFrame()
        print('===cat1_id:{}==='.format(cat1_id_str))

        rst = train_or_forecast(shift_tdf, None, forecast_date.strftime('%Y%m%d'), is_train=True, debug=debug)
        str_model = base64.b64encode(zl.compress(cp.dumps(rst)))
        swdf = pd.DataFrame()
        shift_tdf = pd.DataFrame()
        return model_keyword, str_model
    else:
        swdf = pd.DataFrame(list(line[1][0]), columns=schema_cols)
        swdf = swdf[['dt', 'arranged_cnt', 'is_train'] + raw_features].apply(pd.to_numeric,
                                                                             errors='ignore')  # filtering only using features
        if type(forecast_date) != int:
            forecast_date_int = int(forecast_date.strftime('%Y%m%d'))
        tdf = swdf[(swdf.is_train == 1) & (swdf.dt < forecast_date_int)].reset_index(drop=True)

        tdf.drop_duplicates(subset='sku_id,wh_id,bu_id,dt'.split(','), inplace=True)
        shift_tdf = tdf.groupby('sku_id,wh_id,bu_id'.split(",")).apply(reindex_date)  #
        pdf = swdf[(swdf.is_train == 0) & (swdf.dt >= forecast_date_int)].reset_index(drop=True)
        model = cp.loads(zl.decompress(base64.b64decode(list(line[1][1])[1])))
        if debug:
            with open('objs_f.pkl', 'wb') as f: pickle.dump([shift_tdf, pdf, forecast_date, model], f)
        rst = train_or_forecast(shift_tdf, pdf, forecast_date.strftime('%Y%m%d'), is_train=False, model=model,
                                debug=debug)
        return rst
###########################################xbg case#4  utils###########################################################
