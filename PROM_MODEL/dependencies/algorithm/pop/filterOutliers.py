# encoding=utf-8

"""
Functions for outlier detection and adjustment.
"""

import numpy as np
import pandas as pd
from scipy.stats import lognorm
from scipy.stats import norm
# from statsmodels.nonparametric.smoothers_lowess import lowess

def lognormal_filter(a, q=0.95, min_threshold=10, backfill=0.6):
    """Filter outlier according to lognormal fit.

    Fit a lognomal distribution, then determine the outlier threshold from the parameter p (quantile).

    Args:
        a: array_like, list or numpy array or pandas Series.
        q: lower tail probability, default 0.95.
        min_threshold: Minimal threshold for detecting outlier, compared with threshold from p, default 10.
        backfill: Quantile used to backfill outlier observations, default 0.6.

    Returns:
        Adjusted array (with the same length and data type as the input array).

    Warnings:
        1. The lognormal distribution assumption should be considered.
        2. Require at least 10 non zero values.
    """
    _type = type(a)
    _a = pd.Series(a).copy()

    # Filter out zero values, because the support of lognormal distribution is (0, infinity)
    # nzv stands for non zero values
    _a_nzv = _a[_a > 0]

    # If the number of non zero values less than 10, return the original array
    if _a_nzv.shape[0] < 10:
        return _a

    dist_paras = lognorm.fit(_a_nzv, floc=0)
    outlier_threshold = lognorm.ppf(q, s=dist_paras[0], loc=dist_paras[1], scale=dist_paras[2])
    outlier_threshold = np.max([outlier_threshold, min_threshold])
    outliers = (_a > outlier_threshold)

    if backfill < 1:
        backfill = lognorm.ppf(backfill, s=dist_paras[0], loc=dist_paras[1], scale=dist_paras[2])
    _a[outliers] = backfill

    return _a


def dynamic_ma_filter(a, window_size=7, q=0.95, backfill=0.6):
    """Filter outlier according to moving average mean and sd.

    Moving average and standard deviation are computed from given sliding window size.
    Values above mean+threshold*sd are tagged as outlier.
    Threshold is calculated using standard normal quantiles.

    Args:
        a: array_like, list or numpy array or pandas Series.
        window_size: Size for sliding window.
        q: lower tail probability, default 0.95.
        backfill: Quantile used to backfill outlier observations, default 0.6.

    Returns:
        Adjusted array, format: pandas Series.
    """
    _type = type(a)
    _a = pd.Series(a).copy()

    # window_size check
    try:
        window_size = int(window_size)
    except ValueError:
        raise ValueError("window_size can not be transformed into an integer")

    _a_smooth = _a.rolling(window_size, center=False).mean().fillna(method='bfill')
    residual = _a - _a_smooth
    # compute rolling std
    std = residual.rolling(window_size).std().fillna(method='bfill')

    outliers = _a > (_a_smooth + norm.ppf(q) * std)
    bf = _a_smooth + norm.ppf(backfill) * std
    _a[outliers] = bf[outliers]

    return _a


# def local_smoothing_filter(a, method='lowess', fold_change=3, **kwargs):
#     """Filter outlier according to some local smoothing method and fold change threshold.
#
#     Apply local smoothing method to the original array, then compare the original value with the smoothed value.
#     Values with large fold change are tagged as outliers.
#
#     Args:
#         a: array_like, list or numpy array or pandas Series.
#         method: smoothing method to use, default 'lowess';
#                 for each method, parameters can be passed through key=value
#                     'lowess': frac (default 0.1), it (default 3)
#         fold_change: fold change threshold for outlier detection.
#
#     Returns:
#         Adjusted array (with the same length and data type as the input array).
#     """
#     _type = type(a)
#     _a = pd.Series(a).copy()
#     _index = _a.index
#
#     if method == 'lowess':
#         frac = kwargs.get('frac', 0.1)
#         it = kwargs.get('it', 3)
#         smoothed_data = lowess(endog=_a, exog=np.arange(len(_a)), frac=frac, it=it, return_sorted=False)
#         smoothed_data = pd.Series(smoothed_data, index=_index)
#     else:
#         raise Exception('Smoothing method not supported for now!')
#
#     outlier = _a > (smoothed_data * fold_change)
#     _a[outlier] = smoothed_data[outlier]
#
#     return _a


def quantile_filter(a, q=0.96, backfill=0.8, interpolation='linear'):
    """Filter outlier according to quantile

    Args:
        a: array_like, list or numpy array or pandas Series.
        q: lower tail probability.
        backfill: lower tail probability for outlier backfill.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to
            use when the desired quantile lies between two data points
            ``i < j``:

                * linear: ``i + (j - i) * fraction``, where ``fraction``
                  is the fractional part of the index surrounded by ``i``
                  and ``j``.
                * lower: ``i``.
                * higher: ``j``.
                * nearest: ``i`` or ``j``, whichever is nearest.
                * midpoint: ``(i + j) / 2``.

    Returns:
        Adjusted array (with the same length and data type as the input array).
    """
    _type = type(a)
    _a = pd.Series(a).copy()
    _index = _a.index
    threshold = np.quantile(_a, q, interpolation=interpolation)
    bf = np.quantile(_a, backfill, interpolation=interpolation)
    outliers = _a > threshold
    _a[outliers] = bf

    return _a


def best_coverage_quantile_filter(a, absolute_gap=(2.0, 2.0), relative_gap=(0.5, 0.4), lowest=60):
    """Filter outlier according to best coverage quantile and specified window

    Args:
        a: array_like, list or numpy array or pandas Series.
        absolute_gap: absolute window surrounding best quantile.
        relative_gap: relative window surrounding best quantile.
        lowest: lowest quantile where to start the best search.

    Returns:
        Adjusted array (with the same length and data type as the input array).
    """
    _type = type(a)
    _a = pd.Series(a).copy()
    _index = _a.index

    if len(_a) == 0:
        return _a

    best_quantile = np.NaN
    cover_cnt = np.NaN

    for i in range(lowest, 101):
        quantile = np.percentile(_a, i, interpolation='nearest')
        cnt = 0
        for t in _a:
            if (-absolute_gap[0] <= t - quantile <= absolute_gap[1]) \
                    or (quantile * (1.0 - relative_gap[0]) <= t <= quantile * (1.0 + relative_gap[1])):
                cnt += 1
        if cover_cnt is np.NaN or cnt > cover_cnt:
            cover_cnt = cnt
            best_quantile = quantile

    for i in range(len(_a)):
        t = _a[i]
        if t > (best_quantile + absolute_gap[1]) \
                and (t > best_quantile * (1.0 + relative_gap[1])):
            _a[i] = max(best_quantile + absolute_gap[1], best_quantile * (1.0 + relative_gap[1]))
    return _a


if __name__ == '__main__':
    a = [1, 2, 1.3, 2.5, 6.7, 3, 3, 3, 6, 9, 12]
    b = best_coverage_quantile_filter(a)
    print(a)
    print(b)
