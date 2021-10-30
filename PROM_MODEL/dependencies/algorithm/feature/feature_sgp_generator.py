from functools import reduce
from math import ceil

import numpy as np
import pandas as pd


# 计算 Synthetic Gross Price Temp
def set_temp_sgp(p_df, price_col='net_price', target_col='temp_sgp', interval=7):
    new_df = pd.DataFrame(index=p_df.index)
    p_df_len = len(p_df)
    rst_series = p_df[price_col].groupby(np.arange(p_df_len) // interval).min()
    new_df[target_col] = pd.Series(
        reduce(lambda x, y: x+y, [[val] * interval for val in rst_series])[:p_df_len], index=new_df.index
    )
    return new_df


# 计算 Synthetic Gross Price
def set_sgp(p_df, price_col='sale_price', sgp_col='temp_sgp', target_col='sgp', interval=7):
    new_df = pd.DataFrame(index=p_df.index)
    cell_count = ceil(p_df.index.size / interval)
    index_range_list = np.array_split(p_df.index, cell_count)
    for index_range in index_range_list:
        g = p_df.loc[index_range, :]
        new_df.at[index_range, target_col] = g[price_col].iloc[(g[sgp_col] - g[price_col]).abs().argsort().iloc[0]]
    return new_df


def set_max_col_value(p_df, origin_col_name, target_col_name='max_price'):
    new_df = pd.DataFrame(index=p_df.index)
    new_df[target_col_name] = p_df[origin_col_name].rolling(30, 1).max()
    return new_df


def add_sgp(order_df, sgp_col, window_size=14, interval=7, date_key='cur_date', price_col_name='sale_price'):
    order_df[price_col_name] = order_df['arranged_amt'] / order_df['arranged_cnt']
    order_df['max_price'] = order_df.groupby(['sku_id', 'wh_id', 'bu_id'], group_keys=False, as_index=False).apply(lambda x: set_max_col_value(x, price_col_name))
    order_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    order_df.loc[np.isnan(order_df[price_col_name]) | (order_df[price_col_name] == 0), price_col_name] = order_df.loc[np.isnan(order_df[price_col_name]) | (order_df[price_col_name] == 0), 'max_price']
    order_df.dropna(subset=[price_col_name], inplace=True)

    if len(order_df.index) == 0:
        order_df = pd.DataFrame()
        return
    sku_avg_net_price_day = order_df.loc[:, [date_key, 'sku_id', 'wh_id', 'bu_id', price_col_name]]

    # 计算时间窗口为 window_size 的avg net price
    sku_avg_net_price_day.sort_values(['sku_id', 'wh_id', 'bu_id', 'cur_date'], inplace=True)
    sku_avg_net_price_day['avg_net_real_price_60d'] = sku_avg_net_price_day.groupby(
        ['sku_id', 'wh_id', 'bu_id'], group_keys=False
    )[price_col_name].rolling(window_size, 1).mean().reset_index().set_index('level_3')[price_col_name]

    # 标记 flag day
    sku_avg_net_price_day['is_flag_day'] = sku_avg_net_price_day.apply(lambda x: 1 if np.abs(x[price_col_name] - x['avg_net_real_price_60d']) / x['avg_net_real_price_60d'] > 0.05 else 0, axis=1)

    # 计算 Net Price for calculation
    sku_avg_net_price_day['net_price'] = sku_avg_net_price_day.apply(lambda x: x['avg_net_real_price_60d'] if x['is_flag_day'] else x[price_col_name], axis=1)

    # 计算 Temp SGP
    sku_avg_net_price_day['temp_sgp'] = sku_avg_net_price_day.groupby(['sku_id', 'wh_id', 'bu_id'], group_keys=False, as_index=False).apply(lambda x: set_temp_sgp(x, 'net_price', 'temp_sgp', interval))

    order_df[sgp_col] = 0
    order_df[sgp_col] = sku_avg_net_price_day.groupby(['sku_id', 'wh_id', 'bu_id', 'temp_sgp'], group_keys=False, as_index=False).apply(lambda x: set_sgp(x, price_col_name, 'temp_sgp', 'sgp', interval))


def generate_sgp_feats(lines, schema_cols, window_size=14, interval=7, sgp_col_name='sgp', price_col_name='sale_price'):

    origin_df = pd.DataFrame(list(lines), columns=schema_cols).apply(pd.to_numeric, errors='ignore')
    origin_df.sort_values(['sku_id', 'wh_id', 'bu_id', 'cur_date'], inplace=True)

    add_sgp(origin_df, sgp_col_name, window_size, interval, price_col_name=price_col_name)
    if len(origin_df.index) == 0:
        return None
    rst_schema_cols = schema_cols + [price_col_name, sgp_col_name]
    return origin_df.loc[:, rst_schema_cols].to_records(index=False).tolist()
