import pandas as pd
import datetime

import pandas as pd
import datetime
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

def preprocess(y, fc_dt):
    """
    预测数据预处理
    """
    y = y.sort_index()
    y = y.resample('D').asfreq().bfill().ffill()
    
    # 仅保留预测日期前的数据
    last_valid_day = pd.to_datetime(fc_dt)-pd.Timedelta(days=1)
    return y[:last_valid_day]

def moving_avg(y, fc_dt, ndays = 20, gap_limit = 31):
    """
    加权平均法输出预测结果
    """
    
    assert isinstance(y, pd.Series) and len(y) > 0

    # 过去一段时间无数据不预测
    if (pd.to_datetime(fc_dt) - y.index[-1]).days > gap_limit:
        return None

    y = preprocess(y, fc_dt)
    
    avg_result = (y[-7:].mean() + y[-14:].mean())/2
    
    y_index = pd.date_range(fc_dt, periods = ndays, freq='D')
    return pd.Series([round(avg_result, 1)] * ndays, index=y_index)

def ses(y, fc_dt, ndays = 20, gap_limit = 31):
    """
    使用简单平滑兜底
    """
    assert isinstance(y, pd.Series) and len(y) > 0

    # 过去一段时间无数据不预测
    if (pd.to_datetime(fc_dt) - y.index[-1]).days > gap_limit:
        return None
    
    y = preprocess(y, fc_dt)

    try:
        y_pred = SimpleExpSmoothing(y).fit().forecast()[0]

    except:
        # 异常时使用过去7天平均
        y_pred = y[-7:].mean()

    y_index = pd.date_range(fc_dt, periods = ndays, freq='D')
    return pd.Series([round(y_pred, 1)] * ndays, index=y_index)
    
