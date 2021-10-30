#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------
# File  : longtail_process.py
# Author: liushichang
# Date  : 2021/3/23
# Desc  :
# Contact : liushichang@meituan.com
# ----------------------------------

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class SaleLongtailProcess:
    def __init__(self):
        pass

    @staticmethod
    def get_possion_sp(j, sale_avg):
        x = np.exp(-sale_avg)
        y = 1
        for k in range(1, j + 1):
            y *= sale_avg / k
        return x * y

    @staticmethod
    def get_possion_tp(i, sale_avg):
        tp = 0
        for j in range(0, i+1):
            sp = SaleLongtailProcess.get_possion_sp(j, sale_avg)
            tp += sp
        return tp

    @staticmethod
    def get_possion_val(target_pb, sale_avg):
        for i in range(0, 10000):
            p = SaleLongtailProcess.get_possion_tp(i, sale_avg)
            if p > target_pb:
                return i
        return i

    @staticmethod
    def get_possion_results(df_longtail, pb_val):
        possion_results = []
        target_pb = pb_val
        for index, row in df_longtail.iterrows():
            sale_avg = row['sale_avg']
            days = row['sale_days']
            # 设置数值上限
            if sale_avg > 7:
                sale_avg = 7
            result = SaleLongtailProcess.get_possion_val(target_pb, sale_avg)
            if result < sale_avg:
                result = sale_avg
            possion_results.append([row['bu_id'], row['wh_id'], row['sku_id'], row['cat2_name'], result / days * 7])
        return pd.DataFrame(possion_results, columns=['bu_id', 'wh_id', 'sku_id', 'cat2_name', 'possion_val'])
