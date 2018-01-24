import numpy as np
import pytest
from hypothesis import given

from vai.utils import find_outliers, smoothen
from hypothesis.extra import numpy as nph
from hypothesis import strategies as st


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


class TestSmoothen:
    def test_cannot_send_none(self):
        with pytest.raises(TypeError):
            smoothen(None)

        with pytest.raises(TypeError):
            smoothen(np.zeros(5), None)

        with pytest.raises(TypeError):
            smoothen(np.zeros(5), polyorder=None)

    def test_cannot_send_empty(self):
        with pytest.raises(ValueError):
            smoothen([])

        with pytest.raises(ValueError):
            smoothen(np.array([]))

    @given(st.floats())
    def test_fraction_is_fraction(self, window_fraction):
        if 0 < window_fraction <= 1:
            return

        with pytest.raises(ValueError):
            smoothen(np.zeros(5), window_fraction=window_fraction)

    @given(nph.arrays(nph.floating_dtypes(), nph.array_shapes(max_dims=1, max_side=100)))
    def test_polyorder_cannot_be_greater_than_or_equal_to_data_length(self, data):
        polyorder = np.random.randint(len(data), len(data) * 2)
        with pytest.raises(ValueError):
            smoothen(data, polyorder=polyorder)

    @given(nph.arrays(nph.floating_dtypes(), nph.array_shapes(max_dims=1, min_side=2, max_side=100)), st.floats(0, 1),
           st.integers(0, 100))
    def test_polyorder_cannot_be_greater_than_window_length(self, data, window_fraction, polyorder):
        window_length = int(window_fraction * len(data))
        if window_length % 2 == 0:
            window_length = max(1, window_length - 1)

        if polyorder <= window_length:
            return

        with pytest.raises(ValueError):
            smoothen(data, window_fraction, polyorder)
