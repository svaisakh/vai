import numpy as np
import pytest

from vai.utils import find_outliers


class TestRemoveOutlier:
    def test_data_can_be_1d(self):
        find_outliers(np.zeros(5))

    def test_cannot_send_none(self):
        with pytest.raises(Exception):
            find_outliers(None)

    def test_cannot_send_empty(self):
        with pytest.raises(ValueError):
            find_outliers([])

        with pytest.raises(ValueError):
            find_outliers(np.array([]))

    def test_threshold_not_negative(self):
        with pytest.raises(ValueError):
            find_outliers(np.zeros(5), -1)

    def test_threshold_not_none(self):
        with pytest.raises(ValueError):
            find_outliers(np.zeros(5), None)

    def test_threshold_not_inf(self):
        with pytest.raises(ValueError):
            find_outliers(np.zeros(5), np.inf)

    def test_window_not_negative(self):
        with pytest.raises(ValueError):
            find_outliers(np.zeros(5), window_fraction=-1)

    def test_window_not_greater_than_one(self):
        with pytest.raises(ValueError):
            find_outliers(np.zeros(5), window_fraction=2)
