from numpy.core.defchararray import index
from src.time_series import AutoARIMARegression, BruteForceARIMARegression, SARIMARegression, OverspecifiedError
import unittest
from unittest import mock

import pandas as pd
import datetime as dt
import numpy as np
from statsmodels.iolib import summary

class TestRegression():
    '''Tests for SARImARegression'''

    # Testing strategy:
    # Partition on __init__:
    #   p: 0, 1, >1
    #   d: 0, 1, >1
    #   q: 0, 1, >1
    #   P: 0, 1, >1
    #   D: 0, 1, >1
    #   Q: 0, 1, >1
    #   m: 2, 4, 12
    #   ValueError = None, Raised
    # Partition on fit:
    #   len y: 1, 
    #   # exog vars: 0, 1, >1
    #   OverspecifiedError: Raises, None
    # Partition on predict:
    #   len y: 1, 2, >2
    #   # exogeneous vars: 0, 1, >1
    #   len future X: 0, 1, >1, None
    #   Raises: ValueError, None
    # TODO partition on sliding window

    def check_predictions(self, y, yhats):
        '''
        Asserts the postcondition on yhats.

        - Float64 format
        - No infinite or null values
        - No null indices
        - Same freqstr as y
        - Observation occurs one timestep after y
        '''
        self.assertEqual('float64', yhats.dtype)
        self.assertFalse(np.any(yhats.isnull()))
        self.assertFalse(np.any(np.isinf(yhats)))
        self.assertFalse(np.any(yhats.index.isnull()))

        # TODO refactor to utility
        self.assertEqual(y.index.freqstr, yhats.index.freqstr)
        if len(yhats) >0:
            db_index_delta = yhats.index.min() - y.index.max()
            assert db_index_delta.n == 1

    def check_sliding_forecast(self, y, forecast):
        '''
        Asserts postcondition on sliding forecast.
        
        - Two columns: 'predictions' 'actuals'
        - Last index should be last index of y
        - Same frequency index as y
        - No NaN's or infs
        - No NaN's in index
        - dtype is float64
        '''
        self.assertEqual(2, len(forecast.columns))
        self.assertEqual({'predictions', 'actuals'}, set(forecast.columns))

        # last index is last index of y
        if len(forecast) > 0:
            self.assertEqual(y.index[-1], forecast.index[-1])

        # same freq
        self.assertEqual(y.index.freqstr, forecast.index.freqstr)
        
        # No NaN's or infs
        self.assertFalse(np.any(forecast.isnull()))
        self.assertFalse(np.any(np.isinf(forecast)))

        # No NaN's in index
        self.assertFalse(np.any(forecast.index.isnull()))

        # TODO check dtype of each col
        # self.assertEqual('float64', forecast.dtype)


    # Covers:
    # Partition on __init__:
    #   p=0, d=0, q=0, P=0, D=0, Q=0, m = 12
    # Partition on fit:
    #   len y = 2, # exog vars = 0, OverspecifiedError = None
    # Partition on predict:
    #   n>1, len y=2, # exog vars = 0, len future_X = None, ValueError = None, OverspecifiedError = None
    # Partition on sliding_window_forecast:
    #   len y = 2, # exog vars = 0, window_size = close to 0, OverspecifiedError = None
    def test_no_regressors(self):
        idx = pd.PeriodIndex(freq='Q', year=[2009, 2009], quarter=[1,2])
        y = pd.Series(np.array([1.0, 2.0]),
                        index=idx,
                        dtype='float64',
                        name='endogeneous')
        X = pd.DataFrame(
                        index=idx,
                        dtype='float64',
                        columns=[]
        )

        p_d_q_order = (0,0,0)
        P_D_Q_order = (0,0,0,12)
        model = self.model(p_d_q_order, P_D_Q_order)

        self.assertEqual(p_d_q_order, model.p_d_q_order)
        self.assertEqual(P_D_Q_order, model.P_D_Q_order)
        self.assertTrue(type(model.fit(y, X)) is summary.Summary)

        n=3
        yhat = model.predict(n, y, X)
        self.assertEqual((n,), yhat.shape)
        self.check_predictions(y, yhat)

        with self.assertRaises(OverspecifiedError):
            model.sliding_window_forecast(y, X, window_size=0.001)


    # Covers:
    # Partition on __init__:
    #   p=1, d=0, q=0, P=1, D=0, Q=0, m=2, ValueError = None
    # Partition on fit:
    #   len y >2, # exog vars = 1, OverspecifiedError = None
    # Partition on predict:
    #   n=1, len y >2, # exog vars = 1, len future_X >1, ValueError = None, OverspecifiedError = None
    # Partition on sldiding_window_forecast:
    #   len y >2, # exog vars = 1, window_size = 1, OverspecifiedError = None
    def test_one_exogeneous_var(self):
        n_obs=7
        n_future_obs=5
        idx = pd.period_range(start=pd.Period(year=1999, month=11, freq='M'), periods=n_obs+n_future_obs, freq='M')
        y = pd.Series(np.ones(n_obs),
                    index=idx[0:n_obs],
                    dtype='float64')
        X = pd.DataFrame(np.random.rand(n_obs, 1),
                    index=idx[0:n_obs],
                    dtype='float64')
        future_X = pd.DataFrame(np.random.rand(n_future_obs,1),
                    index=idx[n_obs:],
                    dtype='float64')

        # total order = 8 + 1 x vars + 1 trend
        p_d_q_order = (1,0,0)
        P_D_Q_order = (1,0,0,2)
        model = self.model(p_d_q_order, P_D_Q_order)


        self.assertEqual(p_d_q_order, model.p_d_q_order)
        self.assertEqual(P_D_Q_order, model.P_D_Q_order)
        self.assertTrue(type(model.fit(y, X)) is summary.Summary)

        n=1
        yhat = model.predict(n, y, X, future_X)
        self.assertEqual((n,), yhat.shape)
        self.check_predictions(y, yhat)

        self.check_sliding_forecast(y, model.sliding_window_forecast(y, X, 1.0))

    # Covers:
    # Partition on __init__:
    #   p=1, d>1, q>1, P=1, D=1, Q>1, m=4, ValueError = None
    # Partition on fit:
    #   len y >2, # exog vars >1, OverspecifiedError = Raises
    # Partition on predict:
    #   n=1, len y >2, # exog vars >1, len future_X = 1, ValueError = None, OverspecifiedError = Raises
    # Partition on sliding_window_forecast:
    #   len y > 2, # exog vars >1, window_size = 0 < window_size < 1, OverspecifiedError = raises
    def test_overspecified(self):
        n_obs = self.test_overspecified_n_obs
        n_future_obs = 1
        idx = pd.period_range(start=pd.Period(year=1999, freq='Y'), periods=n_obs+n_future_obs, freq='Y')
        y = pd.Series(np.ones(n_obs),
                        index=idx[:n_obs],
                        dtype='float64')
        X = pd.DataFrame(np.random.rand(n_obs, 2),
                        index=idx[:n_obs],
                        dtype='float64')
        future_X = pd.DataFrame(np.random.rand(1,2),
                        index=idx[n_obs:],
                        dtype='float64')

        # total order = 7 + 2 x vars + 1 trend
        p_d_q_order = (1,2,3)
        P_D_Q_order = (1,1,2,4)
        model = self.model(p_d_q_order, P_D_Q_order)


        self.assertEqual(p_d_q_order, model.p_d_q_order)
        self.assertEqual(P_D_Q_order, model.P_D_Q_order)
        with self.assertRaises(OverspecifiedError):
            model.fit(y,X)

        n=1
        with self.assertRaises(OverspecifiedError):
            model.predict(n, y, X, future_X)

        with self.assertRaises(OverspecifiedError):
            model.sliding_window_forecast(y, X, 0.5)

    # Covers:
    # Partition on __init__:
    #   p=0, d=0, q=0, P=0, D=0, Q=0, m=12, ValueError = None
    # Partition on predict:
    #   n=0, len y > 2, # exog vars = 1, len future_X = 0
    # Partition on sliding_window_forecast:
    #   len y >2, # exog vars = 1, widnow_size = 0 < window_size <1, OverspecifiedError = None
    def test_empty_prediction(self):
        n_obs = 3
        n_future_obs = 0
        idx = pd.period_range(start=pd.Period(year=2012, freq='D'), periods=n_obs+n_future_obs, freq='D')
        y = pd.Series(np.ones(n_obs),
                        index=idx[:n_obs],
                        dtype='float64')
        X = pd.DataFrame(np.random.rand(n_obs,1),
                        index=idx[:n_obs],
                        dtype='float64')
        future_X = pd.DataFrame(np.random.rand(0,1),
                        index=idx[n_obs:],
                        dtype='float64')

        # total order = 7 + 2 x vars + 1 trend
        p_d_q_order = (0,0,0)
        P_D_Q_order = (0,0,0,12)
        model = self.model(p_d_q_order, P_D_Q_order)


        n=0
        yhat = model.predict(n, y, X, future_X)
        self.assertEqual((n,), yhat.shape)
        self.check_predictions(y, yhat)

        n=1
        with self.assertRaises(ValueError):
            model.predict(n,y,X,future_X)

        self.check_sliding_forecast(y, model.sliding_window_forecast(y, X))

    # Glass box testing

    # Covers potential bug where length 1 y is allowed because we don't
    # factor in that a trend is fit.  Can't fit a model with a trend to 
    # one data point
    def test_len_one_y(self):
        idx = pd.PeriodIndex(freq='Q', year=[2009], quarter=[1])
        y = pd.Series(np.array([1.0]),
                        index=idx,
                        dtype='float64',
                        name='endogeneous')
        X = pd.DataFrame(
                        index=idx,
                        dtype='float64',
                        columns=[]
        )

        p_d_q_order = (0,0,0)
        P_D_Q_order = (0,0,0,12)
        model = self.model(p_d_q_order, P_D_Q_order)
        
        with self.assertRaises(OverspecifiedError):
            model.fit(y, X)

        with self.assertRaises(OverspecifiedError):
            model.predict(1, y, X)

class TestSARIMARegression(unittest.TestCase, TestRegression):

    model = SARIMARegression

    # Covers:
    # Partition on __init__:
    #   p>1, d=1, q=1, P>1, D>1, Q=1, m=1
    def test_seasonal_two(self):
        n_obs = 9
        idx = pd.period_range(start=pd.Period(year=2007, month=2, freq='M'), periods=n_obs, freq='M')
        y = pd.Series(data=np.ones(n_obs),
                        index=idx,
                        dtype='float64')
        X = pd.DataFrame(data=np.random.rand(n_obs),
                        index=idx,
                        dtype='float64')

        # order 6 + 1 X var + 1 trend, only need 9 obs
        p_d_q_order = (2,1,1)
        P_D_Q_order = (2,3,1,2)
        with self.assertRaises(ValueError):
            model = self.model(p_d_q_order, P_D_Q_order)

    test_overspecified_n_obs = 10

class TestAutoARIMARegression(unittest.TestCase, TestRegression):

    model = AutoARIMARegression

    test_overspecified_n_obs = 1

    # TODO test_seasonal_two equivalent


BruteForceARIMARegression.determine_opt_order = mock.MagicMock()

# class TestBruteForceARIMARegression(unittest.TestCase, TestRegression):

#     None
