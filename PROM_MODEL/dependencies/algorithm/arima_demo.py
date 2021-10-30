
import statsmodels.api as sm


class ArimaModelDemo:


    def forecast(self,train_data):
        final_iterator = []
        for item in train_data:
            final_iterator.append(item)
        
        model = sm.tsa.statespace.SARIMAX(final_iterator, order = (3, 1, 1),
                                    seasonal_order = (0, 0, 0, 7),
                                    enforce_stationarity = False,
                                    enforce_invertibility = False)
        return model.fit().get_forecast(steps = 3).predicted_mean
		#return iter(model.fit().get_forecast(steps = 3).predicted_mean) ## 输出迭代对象，适用于mapPartitions函数
    












