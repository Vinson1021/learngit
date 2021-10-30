#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import sys
# sys.path.append('/opt/meituan/data-forecast-server/data-python')
# 设置随机性
import numpy as np
import random
import os
seed=2021
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
import datetime
import json
import logging.handlers
import xgboost as xgb
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from xgboost.sklearn import XGBRegressor
from copy import deepcopy
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import sys
sys.path.append('../')
# from util.mysql_crud_util import MysqlCrudUtil
# import transport_util

# logger = logging.getLogger('server.transport.pred_for_wh_day_xgb')

############################################offline model ############################
'''
@ 修改版本: v1.0
@ 修改日期: 2021.07.07
@ 修改人:   songyujia
@ 修改描述: xgb预测3p高销模型
'''


class POPXGBModel:

    def __init__(self, train_file,  fc_dt, management_type=2, label="arranged_cnt", past_promotion_day=10, forecast_step=1):
        self.train_file = train_file
        self.model_name = 'xgb'
        self.indicators = 'pop'
        self.label = label
        self.management_type = management_type
        self.fc_dt = pd.datetime.strptime(fc_dt, "%Y%m%d")
        self.past_promotion_day = past_promotion_day
        self.forecast_step = forecast_step

    def load_data(self):
        df_his = self.train_file
        df_his = self.reduce_mem_usage(df_his)[0]
        #删除冗余特征
        df_his=df_his.dropna(axis=0, subset=["dt"])
        df_his.dt = df_his.dt.apply(lambda x: (str(int(x))))
        df_his['dt'] = pd.to_datetime(df_his['dt'])
        # df_his['dt'] = df_his['dt'].apply(lambda x: pd.to_datetime(str(int(x))))
        del df_his['city_id']
        del df_his['date_id']
        del df_his['cat1_id']
        del df_his['day_abbr']
        del df_his['day_num']
        del df_his['western_festival_name']
        del df_his['available_stock']
        del df_his['hash_wh_sku']
        del df_his['cat2_id']
        del df_his['brand_id']
        del df_his['group_label']
        # del df_his['group_label']
        #仅保留上架数据
        df_his = df_his[(df_his.is_on_shelf == 1)]
        #去重
        df_his = df_his.drop_duplicates(subset=['bu_id', 'wh_id', 'dt', 'sku_id'], keep='first')
        # fixing forecast_step
        df_his = df_his[df_his.dt < self.fc_dt]  # filtering less than fc_dt
        df_pre_pred = df_his[df_his.dt == self.fc_dt + datetime.timedelta(days=-1)]
        for i in range(self.forecast_step):
            print('adding test features ', i, df_his.shape)
            df_pre_pred['dt'] = df_pre_pred['dt'].apply(lambda x: x + datetime.timedelta(days=1))
            df_his = pd.concat([df_his, df_pre_pred], axis=0, ignore_index=True)
        print('df_his.shape is ', df_his.shape, 'df_his.columns is ', df_his.columns)
        return df_his

    def reduce_mem_usage(self,props):
        start_mem_usg = props.memory_usage().sum() / 1024 ** 2
        print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
        NAlist = []  # Keeps track of columns that have missing values filled in.
        for col in props.columns:
            if col=='dt':
                continue
            if props[col].dtype != object:  # Exclude strings

                # Print current column type
                print("******************************")
                print("Column: ", col)
                print("dtype before: ", props[col].dtype)

                # make variables for Int, max and min
                IsInt = False
                mx = props[col].max()
                mn = props[col].min()

                # Integer does not support NA, therefore, NA needs to be filled
                if not np.isfinite(props[col]).all():
                    NAlist.append(col)
                    props[col].fillna(mn - 1, inplace=True)

                    # test if column can be converted to an integer
                asint = props[col].fillna(0).astype(np.int64)
                result = (props[col] - asint)
                result = result.sum()
                if result > -0.01 and result < 0.01:
                    IsInt = True

                # Make Integer/unsigned Integer datatypes
                if IsInt:
                    if mn >= 0:
                        if mx < 255:
                            props[col] = props[col].astype(np.uint8)
                        elif mx < 65535:
                            props[col] = props[col].astype(np.uint16)
                        elif mx < 4294967295:
                            props[col] = props[col].astype(np.uint32)
                        else:
                            props[col] = props[col].astype(np.uint64)
                    else:
                        if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                            props[col] = props[col].astype(np.int8)
                        elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                            props[col] = props[col].astype(np.int16)
                        elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                            props[col] = props[col].astype(np.int32)
                        elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                            props[col] = props[col].astype(np.int64)

                            # Make float datatypes 32 bit
                else:
                    props[col] = props[col].astype(np.float32)

                # Print new column type
                print("dtype after: ", props[col].dtype)
                print("******************************")

        # Print final result
        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = props.memory_usage().sum() / 1024 ** 2
        print("Memory usage is: ", mem_usg, " MB")
        print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
        return props, NAlist

    def feature_engineering(self, df_his):
        new_features = ['bu_id', 'wh_id', 'dt', 'sku_id']#, 'is_holiday', 'is_weekend','day_of_week']  # ,'arranged_price','original_price','platform_original_price']
        lookback_range = [1, 2, 3, 4, 5, 6, 7]
        # df_his=self.reduce_mem_usage(df_his)[0]
        train_test=deepcopy(df_his)

        # train_test['dt'] = pd.to_datetime(train_test['dt'])
        # dummies = pd.get_dummies(train_test['dt'].apply(lambda x:x.dayofweek))
        # w_of_d=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        # new_features.extend(w_of_d)
        # dummies.columns=w_of_d
        # train_test=pd.concat([train_test,dummies],axis=1)
        # train_test = train_test[new_features+['arranged_cnt']]

        #adding arranged_cnt,UV,arranged_amt
        for diff in (lookback_range):
            print('looking back step : ',diff)
            feature_name = 'prev_arranged_cnt_' + str(diff)
            feature_name2 = 'prev_UV_' + str(diff)
            feature_name3 = 'prev_arranged_amt_' + str(diff)
            trainset2 = train_test.copy()
            trainset2.loc[:, 'dt'] += datetime.timedelta(days=diff)
            trainset2.rename(columns={'arranged_cnt': feature_name, \
                                      'view_uv': feature_name2, \
                                      'arranged_amt': feature_name3}, \
                             inplace=True)
            train_test = train_test.merge(trainset2[['bu_id', 'wh_id', 'dt', 'sku_id', \
                                                     feature_name, feature_name2, feature_name3]], \
                                          on=['bu_id', 'wh_id', 'sku_id', 'dt'], \
                                          how='left')
            train_test[feature_name] = train_test[feature_name].fillna(0).astype(int)
            train_test[feature_name2] = train_test[feature_name2].fillna(0).astype(int)
            train_test[feature_name3] = train_test[feature_name3].fillna(0).astype(int)
            new_features.append(feature_name)
            new_features.append(feature_name2)
            new_features.append(feature_name3)
            print(train_test.shape, trainset2.shape)
            del trainset2
        #adding view_outstock_uv_,outstock_uv,intent_uv,order_customer_cnt
        for diff in [1, 2 ,3]:
            feature_name = 'prev_view_outstock_uv_' + str(diff)
            feature_name2 = 'prev_outstock_uv_' + str(diff)
            feature_name3 = 'prev_intent_uv_' + str(diff)
            feature_name4 = 'prev_order_customer_cnt_' + str(diff)
            # feature_name5 = 'prev_arranged_price_' + str(1)
            # feature_name6 = 'prev_original_price_' + str(1)
            # feature_name7 = 'prev_platform_arranged_price_' + str(1)
            # feature_name8 = 'prev_platform_original_price_' + str(1)
            trainset2 = train_test.copy()
            trainset2.loc[:, 'dt'] += datetime.timedelta(days=diff)
            trainset2.rename(columns={'view_outstock_uv': feature_name, \
                                      'outstock_uv': feature_name2, \
                                      'intent_uv': feature_name3, \
                                      'order_customer_cnt': feature_name4}, \
                             #                          'arranged_price': feature_name5,\
                             #                          'original_price': feature_name6,\
                             #                          'platform_arranged_price': feature_name7,\
                             #                          'platform_original_price': feature_name8},\
                             inplace=True)
            train_test = train_test.merge(trainset2[['bu_id', 'wh_id', 'dt', 'sku_id', \
                                                     feature_name, feature_name2, feature_name3, feature_name4]], \
                                          on=['bu_id', 'wh_id', 'sku_id', 'dt'], \
                                          how='left')
            train_test[feature_name] = train_test[feature_name].fillna(0).astype(int)
            train_test[feature_name2] = train_test[feature_name2].fillna(0).astype(int)
            train_test[feature_name3] = train_test[feature_name3].fillna(0).astype(int)
            train_test[feature_name4] = train_test[feature_name4].fillna(0).astype(int)
            # train_test[feature_name5] = train_test[feature_name5].fillna(0).astype(int)
            # train_test[feature_name6] = train_test[feature_name6].fillna(0).astype(int)
            # train_test[feature_name7] = train_test[feature_name7].fillna(0).astype(int)
            # train_test[feature_name8] = train_test[feature_name8].fillna(0).astype(int)
            new_features.append(feature_name)
            new_features.append(feature_name2)
            new_features.append(feature_name3)
            new_features.append(feature_name4)
            # new_features.append(feature_name5)
            # new_features.append(feature_name6)
            # new_features.append(feature_name7)
            # new_features.append(feature_name8)
            del trainset2
        # train_test['dt'] = pd.to_datetime(train_test['dt'])
        cal = calendar()
        holidays = cal.holidays(start=pd.to_datetime('20210501'), end=pd.to_datetime(self.fc_dt.strftime('%Y%m%d')))

        train_test['day_of_week'] = train_test['dt'].apply(lambda x: x.dayofweek + 1)
        train_test['isweekend'] = train_test['day_of_week'].apply(lambda x: 1 if (x == 6 or x == 7) else 0)
        train_test['is_holiday'] = train_test.dt.apply(lambda x: 1 if x in holidays else 0)
        new_features.append('isweekend')
        new_features.append('day_of_week')
        new_features.append('is_holiday')

        train_test = train_test[new_features + ['arranged_cnt']]

        for diff in range(1, 3):
            poi_pre_1 = 'prev_arranged_cnt_' + str(diff)
            poi_pre_2 = 'prev_arranged_cnt_' + str(diff + 1)
            feature_name = 'arranged_diff_' + str(diff)
            train_test[feature_name] = (train_test[poi_pre_1] - train_test[poi_pre_2])
            new_features.append(feature_name)
        # del train_test['last_7sum']
        train_test = train_test[new_features + ['arranged_cnt']]
        train_test = train_test.fillna(0)
        return train_test, new_features



    def train_forecast(self):
        '''

        '''
        print('syj training xgb fc_dt is :', self.fc_dt)
        df_his= self.load_data()
        ## adding features
        train_test, new_features = self.feature_engineering(df_his)
        print('syj training xgb feature_engineering is :', len(new_features), new_features)

        ### dt 没啥用
        if train_test['dt'].dtypes != 'int64':
            train_test['dt'] = train_test.dt.apply(lambda x: int(float((x.strftime('%Y%m%d')))))

        train_part = train_test[(train_test.dt < int(self.fc_dt.strftime('%Y%m%d')))].reset_index(drop=True)
        test_part = train_test[(train_test.dt == int(self.fc_dt.strftime('%Y%m%d')))].reset_index(drop=True)
        new_features.remove('bu_id')  #
        new_features.remove('wh_id')
        new_features.remove('dt')
        new_features.remove('sku_id')
        trainx = train_part[new_features]
        trainy = train_part[self.label]
        testx = test_part[new_features]
        df = test_part.iloc[:, 0:4]
        model = xgb.XGBRegressor(objective='reg:squarederror', num_round=1000, max_depth=7, min_child_weight=0.5,
                                 subsample=1, \
                                 eta=0.05, seed=1, alpha=2, colsample_bytree=1, gamma=2)
        # model = xgb.XGBRegressor(objective=asy_mse,max_depth = 11, min_child_weight=0.5, subsample = 1, eta = 0.05, num_round = 1000, seed = 1)
        model.fit(trainx, trainy, eval_metric='rmse')
        preds = model.predict(testx)
        # np.abs(testy-preds).sum()/preds.sum()
        ### construct final output
        result = pd.DataFrame([preds]).T
        result = result.rename(columns={0:'today_fc_cnt'})
        # result['diff'] = np.abs(result.sales_pred - result.real_value)
        # result['one_mape'] = result.apply(lambda x: x['diff'] / x.real_value if x.real_value > 0 else x['diff'], axis=1)
        result = pd.concat([df, result], axis=1)
        result.bu_id=result['bu_id'].apply(lambda x: int(x))
        result['dt'] = result['dt'].apply(lambda x: str(int(x)))

        result.rename(columns={'dt':'fc_dt'},inplace=True)
        result['daily_fc_cnt'] = result.apply(lambda row: json.dumps({row['fc_dt']:row['today_fc_cnt']}), axis=1)

        return result


        # # return result
        # if len(result) > 0:
        #     logger.info('model 将要插入以下数据' + str(result[:2]))
        #     transport_util.insert_or_update_new(result)


if __name__ == '__main__':
    # print(sys.argv)
    train_file_path = '~/Downloads/3p_sku_UV_0630.txt'
    fc_dt = '20210701'
    addrFore = POPXGBModel(train_file_path, fc_dt=fc_dt, forecast_step=1)
    result = addrFore.train_forecast()
    result.to_csv('~/Downloads/result.csv')
    print(result)
    # print('result is :', result)

## train_file_path
"""select base.*,
       sh.cat2_id,
       sh.brand_id,
       UV.view_uv,
       UV.view_outstock_uv,
       UV.outstock_uv,
       UV.intent_uv,
       UV.order_customer_cnt,
       UV.arranged_price,
       UV.original_price,
       UV.platform_arranged_price,
       UV.platform_original_price

  from (
        select sales.*
          from (
                select *
                  from mart_caterb2b_forecast.app_sale_bu_wh_sku_3p_day_input
                 where dt>='20210501'
               ) sales
          join (
                select distinct bu_id,
                       wh_id,
                       sku_id
                  from mart_caterb2b_forecast.app_sale_bu_wh_sku_3p_day_input
                 where dt>='20210501'
                   and (arranged_cnt>0 or is_on_shelf>0)
               ) items
            on sales.bu_id = items.bu_id
           and sales.wh_id = items.wh_id
           and sales.sku_id = items.sku_id
       )base
  join (
        select bu_id,
               wh_id,
               sku_id,
               avg(arranged_cnt) AS avg_cnt
          from mart_caterb2b_forecast.app_sale_bu_wh_sku_3p_day_input
         where dt>='20210501'
         group by bu_id,
                  wh_id,
                  sku_id
        having avg_cnt>5
       ) filter
    on filter.bu_id = base.bu_id
   and filter.wh_id = base.wh_id
   and filter.sku_id = base.sku_id join(
        select bu_id,
               sku_id,
               dt,
               view_outstock_uv,
               outstock_uv,
               intent_uv,
               order_customer_cnt,
               arranged_price,
               original_price,
               platform_arranged_price,
               platform_original_price,
               view_uv
          from mart_caterb2b.total_prod_3p_city_bu_sku_day
         where dt>='20210501'
       ) UV
    on base.bu_id=UV.bu_id
   and base.sku_id=UV.sku_id
   and base.dt=UV.dt
  JOIN mart_caterb2b.dim_caterb2b_sku_his sh
    ON base.sku_id = sh.sku_id
   and base.dt = sh.dt
"""
