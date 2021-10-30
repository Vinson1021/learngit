import statsmodels.api as sm
import numpy as np
import pandas as pd

# auto_arima所需库
from pyramid.arima import auto_arima

# prophet所需库
from fbprophet import Prophet

from statsmodels.tsa.holtwinters import ExponentialSmoothing

import logging

logger = logging.getLogger('fbprophet')
logger.setLevel(logging.WARN)

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)

logger = logging.getLogger('server.timeseries')

class TimeSeriesForecastHour:
    
    def __init__(self, y, n_steps):

        assert len(y) > 0

        self.y = y.copy()
        self.n_steps = n_steps

        # 异常时返回过去24h的平均值
        self.y_pred_exception = [self.y[-24:].mean()] * self.n_steps


    def preprocess(self):
        """
        剔除波动过大的异常值
        """

        # 以小时为采样间隔
        self.y = self.y.resample('H').mean()

        # 如果有空值使用后一值填充
        if self.y.isnull().sum() > 0:
            print('序列中包含空值, 使用bfill填充')
            self.y = self.y.bfill()
            print(self.y)
            
    def auto_arima(self):
        """
        自动arima, 指定pdq范围即可
        """
        try:
            stepwise_model = auto_arima(self.y, start_p=0, start_d =0, start_q=0,
                                        max_p=3, max_q=3, max_d = 3,
                                        seasonal = True,
                                        m = 24, start_P = 0, start_D=0, start_Q=0,
                                        max_P = 2, max_D = 2, max_Q = 2,
                                        trace=False, error_action='ignore', 
                                        suppress_warnings = True, 
                                        stepwise = True   # 如果设置为False会遍历很慢，当前参数无响应
                                       )
            # fit & predict stepwise_model.fit(y)
            y_predict = stepwise_model.fit_predict(self.y, n_periods = self.n_steps)

#             if (y_predict < 0).sum() > 0:
#                 raise RuntimeError('Auto_arima get negative or None results')
  
        except Exception as e:
            print(e)
            logger.info(e)
            # 数据量少时会报异常
            y_predict = self.y_pred_exception
              
        finally:
            # 返回包含所预测天数的Series结构, 与别的模型不同，auto_arima返回的为数组
            y_index = pd.date_range(self.y.index[-1] + pd.to_timedelta(1, 'H'), periods=self.n_steps, freq='H')
            return pd.Series(y_predict, index = y_index)


    def prophet(self, n_changepoints = 20, changepoint_prior_scale = 30):
        """
        调用prophet预测n_steps步，输入为Series类型，且以日期为index
        """
        assert isinstance(self.y, pd.core.series.Series)
        assert isinstance(self.y.index, pd.core.indexes.datetimes.DatetimeIndex)
        
        # 将序列按prophet要求重命名，且转换为dataframe结构
        y_prophet = self.y.copy()
        y_prophet.index.rename('ds', inplace = True)
        y_prophet = y_prophet.reset_index(name = 'y')
            
        try:
            
            model = Prophet(growth = 'linear',
                            n_changepoints = n_changepoints, 
                            changepoint_prior_scale = changepoint_prior_scale,
                            weekly_seasonality = True,
                            yearly_seasonality = False,
                            daily_seasonality = True)

            model.fit(y_prophet)

            # 构造预测时段结构
            future_dates = model.make_future_dataframe(periods = self.n_steps, freq='H')

            forecast = model.predict(future_dates)
            # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            y_predict = forecast[['ds', 'yhat']].set_index('ds')['yhat'][-self.n_steps:]
            
#             # 数据少时prophet有时不会抛异常，但会计算出负数
#             if y_predict.isnull().sum() > 0 or (y_predict < 0).sum() > 0:
#                 raise RuntimeError('Prophet get negative or None results')

        except Exception as e:
            # 数据量少时会报异常
            # 写出具体异常情况
            # 过去一周的平均值作为兜底值
            print(e)
            logger.info(e)
            y_predict = self.y_pred_exception
            y_index = pd.date_range(self.y.index[-1] + pd.to_timedelta(1, 'H'), periods=self.n_steps, freq='H')
            y_predict = pd.Series(y_predict, index = y_index)

        finally:
            return y_predict

    def holtwinters(self):
        """
        调用Holt-Winters预测
        """
        
        try:
            model = ExponentialSmoothing(self.y, 
                                     trend='add',    # 'mul' 
                                     # damped=True, 
                                     seasonal='add',  #'mul'
                                     seasonal_periods = 24
                                    ).fit()  # use_boxcox=True效果不好
            y_predict = model.forecast(self.n_steps)

#             if y_predict.isnull().sum() > 0 or (y_predict < 0).sum() > 0:
#                 raise RuntimeError('holtwinters get negative or None results')
            
        except Exception as e:
            # 数据量少时会报异常
            print(e)
            logger.info(e)
            y_predict = self.y_pred_exception
            y_index = pd.date_range(self.y.index[-1] + pd.to_timedelta(1, 'H'), periods = self.n_steps, freq='H')
            y_predict = pd.Series(y_predict, index = y_index)
        
        finally:
            return y_predict


    def get_predict(self, model = 'auto_arima'):
        """
        调用指定模型完成预测
        """
        # 数据预处理
        self.preprocess()

        if model == 'auto_arima':
            return self.auto_arima()
        elif model == 'prophet':
            return self.prophet()
        elif model == 'holtwinters':
            pass
            return self.holtwinters()
        else:
            raise RuntimeError('model %s is not supported'%model)

if __name__ == '__main__':

    s = pd.Series(range(50), index = pd.date_range('20200301', periods=50, freq='H'))
    print(s)

    ts_forecast = TimeSeriesForecastHour(s, n_steps = 10)
    
    
    print('------------auto_arima------------')
    print(ts_forecast.get_predict('auto_arima'))

    print('------------holtwinters------------')
    print(ts_forecast.get_predict('holtwinters'))

#     print('------------prophet------------')
#     print(ts_forecast.get_predict('prophet'))

