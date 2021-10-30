#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import logging.handlers
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf
# tf.set_random_seed(2020)
import tensorflow_probability as tfp
from tensorflow_probability import sts


warnings.filterwarnings("ignore")  # specify to ignore warning messages


class STSModel:

    def __init__(self, training_data, forecast_step=8, trend=True, seasonal_y1=False, seasonal_y2=False, seasonal_y7=True):
        self.training_data = training_data
        self.forecast_step = forecast_step
        self.trend = trend
        self.seasonal_y1 = seasonal_y1
        self.seasonal_y2 = seasonal_y2
        self.seasonal_y7 = seasonal_y7


    def build_model(self):
        observed_time_series = self.training_data
        build_model_ls = []
        if self.trend:
            trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
            build_model_ls.append(trend)
        if self.seasonal_y1:
            seasonal_y1 = tfp.sts.Seasonal(num_seasons=1, observed_time_series=observed_time_series)
            build_model_ls.append(seasonal_y1)
        if self.seasonal_y2:
            seasonal_y2 = tfp.sts.Seasonal(num_seasons=2, observed_time_series=observed_time_series)
            build_model_ls.append(seasonal_y2)
        if self.seasonal_y7:
            seasonal_y7 = tfp.sts.Seasonal(num_seasons=7, observed_time_series=observed_time_series)
            build_model_ls.append(seasonal_y7)

        model = tfp.sts.Sum(build_model_ls, observed_time_series=observed_time_series)
        return model

    def sts_train_forecast(self):
        training_data = self.training_data
        tf.reset_default_graph()
        with tf.Session() as sess:
            model = self.build_model()
            with tf.variable_scope('sts_elbo', reuse=tf.AUTO_REUSE):
                variational_loss, variational_distributions = tfp.sts.build_factored_variational_loss(
                    model, observed_time_series=training_data)
            with tf.variable_scope('sts_elbo', reuse=tf.AUTO_REUSE):
                train_op = tf.train.AdamOptimizer(0.1).minimize(variational_loss)

            sess.run(tf.global_variables_initializer())

            for step in range(350):
                _, loss_ = sess.run((train_op, variational_loss))
                if step % 50 == 0:
                    print("step {} loss {}".format(step, loss_))

            forecast_mean = np.zeros([training_data.shape[0], self.forecast_step])
            num = 6
            for sample_step in range(num):
                posterior_samples_ = sess.run({param_name: q.sample(1)
                                               for param_name, q in variational_distributions.items()})
                forecast_dist = tfp.sts.forecast(model, observed_time_series=training_data,
                                                 parameter_samples=posterior_samples_,
                                                 num_steps_forecast=self.forecast_step)
                forecast_mean_single = sess.run((forecast_dist.mean()[:][..., 0]))
                forecast_mean = forecast_mean + forecast_mean_single
            forecast_mean = forecast_mean / num
            return forecast_mean.round()
