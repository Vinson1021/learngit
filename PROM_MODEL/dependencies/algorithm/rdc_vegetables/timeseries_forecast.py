import pandas as pd
import datetime
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)


from fbprophet import Prophet
import logging
logger = logging.getLogger('fbprophet')
logger.setLevel(logging.WARN)


def holtwinters(y, fc_dt, ndays = 20, lack_day_limit = 10, use_SES = False):
    """
    功能：使用holtwinters预测未来ndays天的
         fc_dt: 预测起始第一天
         ndays: 预测天数
         blank_limit: 允许在预测起始天前缺失数据的天数
    """

    
    assert isinstance(y, pd.Series)
    
    # 仅保留预测日期前的数据
    last_valid_day = datetime.datetime.strptime(fc_dt, "%Y%m%d")-datetime.timedelta(days=1)
    y = y[:last_valid_day]
    
    # 过滤大促结果
    y = y[y < y.mean()*3]
    
    # 将小于平均值1/3的结果过滤
    # s = s[s > s.mean()/3]
    #     # 将数据填充到预测前一天, 不应该这样填充，如果中间缺少天数比较多时不合适
    #     if last_valid_day != s.index[-1]:
    #         s[last_valid_day] = None
    
    # 预测结果时间index
    fc_dt_index = pd.date_range(fc_dt, periods = ndays, freq='D')
    # fc_dt_index = datetime.datetime.strptime(fc_dt, "%Y%m%d")
    
    if len(y) == 0:
        return None
    else:
        y = y.resample('D').asfreq().bfill().ffill()
    
    if len(y) == 0:
        # 只有测试时才会出现这种情况
        return None
    
    elif len(y) == 1:
        return pd.Series([y[0]] * ndays, index = fc_dt_index)
    
    else:
        # 计算出最近缺失天数
        lack_days = (last_valid_day - y.index[-1]).days
        
        # 缺失天数过多，不宜预测步长过长
        horizon = min(lack_days, lack_day_limit) + ndays


        if use_SES:

            # 使用简单指数平滑
            try:
                y_predict = SimpleExpSmoothing(y).fit().forecast(horizon)

            except:
                # 异常时使用过去7天平均
                y_predict = pd.Series([y[-7:].mean()] * ndays, index = fc_dt_index)

            finally:
                y_predict[y_predict < 0] = 0
                y_predict = round(pd.Series(y_predict[-ndays:].values, index = fc_dt_index), 2)
                
        else:

            # Holtwinters
            try:
                # 数据少于两个周期时可能报错
                y_predict = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=7).fit()\
                .forecast(horizon)

            except:
                # 异常时使用过去7天平均
                y_predict = pd.Series([y[-7:].mean()] * ndays, index = fc_dt_index)       

            finally:
                y_predict[y_predict < 0] = 0
                y_predict = round(pd.Series(y_predict[-ndays:].values, index = fc_dt_index), 2)
                
        return y_predict


def prophet(y, fc_dt, ndays = 20, lack_day_limit = 10, x_regressor_list = []):
    """
    输入预测变量series，以及外部变量名（含有外来值）
    """

    assert isinstance(y, pd.Series)
    
    y = y.copy()
    
    # 仅保留预测日期前的数据
    last_valid_day = datetime.datetime.strptime(fc_dt, "%Y%m%d")-datetime.timedelta(days=1)
    y = y[:last_valid_day]
    
    if len(y) == 0:
        return None
    
    # 预测结果时间index
    fc_dt_index = pd.date_range(fc_dt, periods = ndays, freq='D')
    
    # 计算出最近缺失天数
    lack_days = (last_valid_day - y.index[-1]).days
        
    # 缺失天数过多，不宜预测步长过长
    horizon = min(lack_days, lack_day_limit) + ndays
    
    # prophet允许缺失值
    # s1 = s.resample('D').asfreq().interpolate(method='linear')
    
    y.index.rename('ds', inplace = True)
    y = y.reset_index(name = 'y')
    
    for i, x_regressor in enumerate(x_regressor_list):
        # 添加外部变量，且外部变量不可为Null
        y['regressor_%d'%i] = x_regressor[y.ds].values
        y['regressor_%d'%i] = y['regressor_%d'%i].ffill().bfill()

    model = Prophet(growth = 'linear',
                    # seasonality_mode = 'multiplicative',
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    daily_seasonality = False)

    # 添加外部变量
    for i, x_regressor in enumerate(x_regressor_list):
        model.add_regressor('regressor_%d'%i)
    
    model.fit(y)

    # 构造预测时段结构
    future_dates = model.make_future_dataframe(periods = horizon, freq='D')
    
    for i, x_regressor in enumerate(x_regressor_list):
        future_dates['regressor_%d'%i] = x_regressor[future_dates.ds].values
        future_dates['regressor_%d'%i] = future_dates['regressor_%d'%i].ffill().bfill()
        
    forecast = model.predict(future_dates)
    
    y_predict = forecast[['ds', 'yhat']].set_index('ds')['yhat'][-ndays:]
    y_predict[y_predict < 0] = 0
    
    return round(pd.Series(y_predict[-ndays:].values, index = fc_dt_index), 2)

    
