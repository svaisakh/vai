import warnings

import numpy as np
from scipy.signal import savgol_filter


def find_outliers(data, threshold=3.5, window_fraction=0.05):
    """Based on http://www.itl.nist.gov/div898/handbook/eda/section3
    /eda35h.htm """

    def __handle_args():
        if len(data) == 0:
            raise ValueError('data is empty!')
        if len(data) < 3:
            return np.array([False] * len(data))
        if len(data.shape) == 1:
            return find_outliers(np.expand_dims(data, -1), threshold,
                                 window_fraction)
        if threshold is None:
            raise ValueError('threshold cannot be None')
        if threshold < 0:
            raise ValueError(
                'threshold should be non negative but got {}'.format(
                    threshold))
        elif np.isinf(threshold) or np.isnan(threshold):
            raise ValueError(
                'threshold should be a finite number but got {}'.format(
                    threshold))

        if window_fraction < 0 or window_fraction > 1:
            raise ValueError(
                'window_fraction should be a fraction (duh!). But got '
                '{}'.format(
                    window_fraction))
        elif np.isinf(window_fraction) or np.isnan(window_fraction):
            raise ValueError(
                'threshold should be a finite number but got {}'.format(
                    threshold))

    arg_err = __handle_args()
    if arg_err is not None:
        return arg_err

    # Subdivide data into small windows
    window_length = max(int(len(data) * window_fraction),
                        3) if window_fraction is not None else 3
    divide_ids = np.arange(window_length, len(data), window_length)

    split_data = np.split(data, divide_ids)

    def _find_outliers(x):
        outlier_factor = 0.6745

        median = np.median(x, axis=0)
        distances = np.linalg.norm(x - median, axis=-1)
        median_deviation = np.median(distances)

        # No deviation. All values are same. No outlier
        if median_deviation == 0:
            return np.array([False] * len(x))
        modified_z_scores = outlier_factor * distances / median_deviation

        outlier_mask = modified_z_scores > threshold

        return outlier_mask

    return np.concatenate([_find_outliers(d) for d in split_data])


def smoothen(data, window_fraction=0.3, **kwargs):
    order = kwargs['order'] if 'order' in kwargs.keys() else 3

    def __handle_args():
        if type(data) is not np.ndarray and type(data) is not list:
            raise TypeError('data needs to be a list or numpy array. Got {}'.format(type(data)))
        if len(data) == 0:
            raise ValueError('data is empty!')
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError('some of the data is either nan or inf')

        if type(window_fraction) is not float:
            raise TypeError('window_fraction should be a fraction (duh!). But got {}'.format(type(window_fraction)))
        if window_fraction < 0 or window_fraction > 1:
            raise ValueError('window_fraction should be a fraction (duh!). But got {}'.format(window_fraction))
        if np.isinf(window_fraction) or np.isnan(window_fraction):
            raise ValueError('window_fraction should be a finite number but got {}'.format(window_fraction))

        min_frac = order / len(data) if order % 2 == 0 else (order + 1) / len(data)
        if window_fraction <= min_frac:
            warnings.warn('window_fraction ({}) too low for polyorder ({}) and length ({}) of data. The minimum '
                          'possible allowed is {}.\nReturning raw data'.format(window_fraction, order, len(data),
                                                                               min_frac), RuntimeWarning)
            return data

        if type(order) is not int:
            raise TypeError('polyorder needs to be a non-negative integer but got {}'.format(type(order)))
        if order < 0:
            raise ValueError('polyorder needs to be a non-negative integer but got {}'.format(order))

    arg_err = __handle_args()
    if arg_err is not None:
        return arg_err

    window_length = int(len(data) * window_fraction)
    # savgol_filter needs an odd window_length
    if window_length % 2 == 0:
        window_length = max(window_length - 1, 1)

    return savgol_filter(data, window_length, order)
