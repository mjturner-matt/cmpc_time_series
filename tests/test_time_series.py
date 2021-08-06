from numpy.core.defchararray import index
from src.time_series import AutoARIMARegression, BruteForceARIMARegression, SARIMARegression, OverspecifiedError
import unittest
from unittest import mock
from unittest.mock import call, patch

import pandas as pd
import datetime as dt
import numpy as np
from statsmodels.iolib import summary

from tests.test_utils import TestUtils

import timeout_decorator

class TestRegression(TestUtils):
    '''Tests for all regression models.'''

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
    #   len y: 1, 2, >2
    #   # exog vars: 0, 1, >1
    #   OverspecifiedError: Raises, None
    # Partition on predict:
    #   len y: 1, 2, >2
    #   # exogeneous vars: 0, 1, >1
    #   len future X: 0, 1, >1, None
    #   Raises: ValueError, None
    # Partition on sliding_window_forecast:
    #   len y: 1, 2, >2
    #   # exogeneous vars: 0, 1, >1
    #   window_size = close to 0, 0 < window_size <1, 1
    #   OverspecifiedError: Raises, None

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
    #   p=0, d=0, q=0, P=0, D=0, Q=0, m = 12, ValueError = None
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
        model = self.create_model(y, X, p_d_q_order, P_D_Q_order)

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
        model = self.create_model(y, X, p_d_q_order, P_D_Q_order)

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
        model = self.create_model(y, X, p_d_q_order, P_D_Q_order)

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
        model = self.create_model(y, X, p_d_q_order, P_D_Q_order)

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
        model = self.create_model(y, X, p_d_q_order, P_D_Q_order)
        
        with self.assertRaises(OverspecifiedError):
            model.fit(y, X)

        with self.assertRaises(OverspecifiedError):
            model.predict(1, y, X)

    # Test cases resulting from bugs
    def test_no_exog_sliding_window(self):
        n_obs = 11
        idx = pd.period_range(start=pd.Period(year=2020, month=11, freq='M'), periods=n_obs, freq='M')
        y = pd.Series(np.random.rand(n_obs),
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
        model = self.create_model(y, X, p_d_q_order, P_D_Q_order)

        self.check_sliding_forecast(y, model.sliding_window_forecast(y, X, window_size=0.8))

# TODO test SARIMA and Brute

class TestSARIMARegression(unittest.TestCase, TestRegression):

    model = SARIMARegression

    def create_model(self, y, X, p_d_q_order, P_D_Q_order):
        '''Creates a model object for this test class.'''
        return SARIMARegression(p_d_q_order, P_D_Q_order)

    # Covers:
    # Partition on __init__:
    #   p>1, d=1, q=1, P>1, D>1, Q=1, m=1, ValueError = Raises
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
            model = self.create_model(y, X, p_d_q_order, P_D_Q_order)

    test_overspecified_n_obs = 10

    # Error testing

    # Error with slididng window not returning a value
    def test_sliding_window_len_3_y(self):
        n_obs = 3
        idx = pd.period_range(start=pd.Period(year=2003, month=2, freq='M'), periods=n_obs, freq='M')
        y = pd.Series(data=np.random.rand(n_obs),
                        index=idx,
                        dtype='float64')
        X = pd.DataFrame(
                        index=idx,
                        dtype='float64')

        p_d_q_order = (0,0,0)
        P_D_Q_order = (0,0,0,2)

        model = self.create_model(y,X,p_d_q_order, P_D_Q_order)

        yhats, actuals = model._sliding_window(y, None, p_d_q_order, P_D_Q_order, 2)
        self.assertEqual(1, len(yhats))
        self.assertEqual(1, len(actuals))

class TestAutoARIMARegression(unittest.TestCase, TestRegression):

    model = AutoARIMARegression

    def create_model(self, y, X, p_d_q_order, P_D_Q_order):
        '''Creates a model object for this test class.'''
        return AutoARIMARegression(p_d_q_order, P_D_Q_order)

    test_overspecified_n_obs = 1

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
        model = self.create_model(y, X, p_d_q_order, P_D_Q_order)

class TestBruteForceARIMARegression(unittest.TestCase, TestRegression):

    # relies on assumption that determine_opt_order is only run once,
    # so we only have to patch it in here.
    # Mock means that __init__ will raise no errors, so works for other test cases above.
    @patch('src.time_series.BruteForceARIMARegression.determine_opt_order')
    def create_model(self, y, X, p_d_q_order, P_D_Q_order, mock_determine_opt_order):
        '''Creates a model object for this test class.'''
        p,d,q = p_d_q_order
        P,D,Q,m = P_D_Q_order
        order_arg_ranges = (range(0,p+1), #p
                            range(0,d+1), #d
                            range(0,q+1), #q
                            range(0,P+1), #P
                            range(0,D+1), #D
                            range(0,Q+1), #Q,
                            range(m, m+1)) #m
        
        mock_determine_opt_order.return_value = (p,d,q,P,D,Q,m)

        model = BruteForceARIMARegression(y, X, order_arg_ranges)

        mock_determine_opt_order.assert_called_once()
        call_y, call_X, call_training_window_size, call_order_arg_ranges = mock_determine_opt_order.call_args.args

        self.assert_equal_series(y, call_y)
        self.assert_equal_dataframes(X, call_X)
        self.assertTrue(len(y) >= call_training_window_size)
        self.assertTrue(0 <= call_training_window_size)
        self.assertEqual(order_arg_ranges, call_order_arg_ranges)

        return model

    test_overspecified_n_obs = 1

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
        model = self.create_model(y, X, p_d_q_order, P_D_Q_order)

    # Testing strategy
    # Partitions on determine_opt_order:
    #   len y: 2, 3, >3  - only possible since otherwise model would be overspecified with trend which violates precondition
    #   # egos vars: 0, 1, >1
    #   train_size: 1, 1<train_size<len(y)-1, len(y)-1
    #   p start: 0, >0
    #   p end: 1, >1
    #   p interval len: 1, >1
    #   d start: 0, >0
    #   d end: 1, >1
    #   d interval len: 1, >1
    #   q start: 0, >0
    #   q end: 1, >1
    #   q interval len: 1, >1
    #   P start: 0, >0
    #   P end: 1, >1
    #   P interval len: 1, >1
    #   D start: 0, >0
    #   D end: 1, >1
    #   D interval len: 1, >1
    #   Q start: 0, >0
    #   Q end: 1, >1
    #   Q interval len: 1, >1
    #   m start: 2 >2
    #   m end: 3, >3
    #   m interval len: 1, >1
    #   OverspecifiedError: Raises, None
    #   ValueError: Raises, None

    # Covers:
    #   len y = 3, # exog vars = 0
    #   p start = 0, p end = 1, p interval len = 1
    #   d start = 0, d end = 1, d interval len = 1
    #   q start = 0, q end = 1, q interval len = 1
    #   P start = 0, P end = 1, P interval len = 1
    #   D start = 0, D end = 1, D interval len = 1
    #   Q start = 0, Q end = 1, Q interval len = 1
    #   m start = 2, m end = 3, m interval len = 1
    #   OverspecifiedError = None, ValueError = None
    def test_simple_one_solution(self):
        n_obs = 3
        idx = pd.period_range(start=pd.Period(year=2003, month=2, freq='M'), periods=n_obs, freq='M')
        y = pd.Series(data=np.random.rand(n_obs),
                        index=idx,
                        dtype='float64')
        X = pd.DataFrame(
                        index=idx,
                        dtype='float64')
        p_start, d_start, q_start = (0,0,0)
        P_start, D_start, Q_start, m_start = (0,0,0,2)
        p,d,q = (1,1,1)
        P,D,Q,m = (1,1,1,3)
        order_arg_ranges = (range(p_start, p),
                            range(d_start, d),
                            range(q_start, q),
                            range(P_start, P),
                            range(D_start, D),
                            range(Q_start, Q),
                            range(m_start, m))

        model = BruteForceARIMARegression(y, X, order_arg_ranges)

        self.assertEqual((p_start, d_start, q_start), model.p_d_q_order)
        self.assertEqual((P_start, D_start, Q_start, m_start), model.P_D_Q_order)

    # Covers:
    #   len y >2, # exog vars >1,
    #   p start >0, p end >1, p interval len >1,
    #   d start >0, d end >1, d interval len >1,
    #   q start >0, q end >1, q interval len >1,
    #   P start >0, P end >1, P interval len >1,
    #   D start >0, D end >1, P interval len >1,
    #   Q start >0, Q end >1, Q interval len >1,
    #   m start >2, m end >3, m interval len >1
    #   OverspecifiedError = Raises, ValueError = None
    @timeout_decorator.timeout(1)
    def test_all_order_args_overspecified(self):
        n_obs = 18
        idx = pd.period_range(start=pd.Period(year=2003, month=2, freq='M'), periods=n_obs, freq='M')
        y = pd.Series(data=np.random.rand(n_obs),
                        index=idx,
                        dtype='float64')
        X = pd.DataFrame(data=np.random.rand(n_obs,10),
                        index=idx,
                        dtype='float64')
        # min total order = 1 + 1 + 2 + 3 + 1 trend + 10 X = 18
        p_start, d_start, q_start = (1,2,1)
        P_start, D_start, Q_start, m_start = (2,1,3,6)
        p,d,q = (3,5,4)
        P,D,Q,m = (10,14,12,100)
        order_arg_ranges = (range(p_start, p),
                            range(d_start, d),
                            range(q_start, q),
                            range(P_start, P),
                            range(D_start, D),
                            range(Q_start, Q),
                            range(m_start, m))

        with self.assertRaises(OverspecifiedError):
            model = BruteForceARIMARegression(y, X, order_arg_ranges)

    # Covers:
    #   len y >3, # exog vars = 1
    #   p start  >0 , p end  > 1, p interval len = 1
    #   d start = 0, d end = 1, d interval len = 1
    #   q start = 0, q end = 1, q interval len = 1
    #   P start > 0, P end > 1, P interval len = 1
    #   D start = 0, D end = 1, D interval len = 1
    #   Q start = 0, Q end = 1, Q interval len = 1
    #   m start = 2, m end = 3, m interval len = 1
    #   OverspecifiedError = None, ValueError = None
    def test_all_order_args_repeat_lags(self):
        n_obs = 6
        idx = pd.period_range(start=pd.Period(year=2003, month=2, freq='M'), periods=n_obs, freq='M')
        y = pd.Series(data=np.random.rand(n_obs),
                        index=idx,
                        dtype='float64')
        X = pd.DataFrame(np.random.rand(n_obs,1),
                        index=idx,
                        dtype='float64')
        # min total order = 1 + 1 + 1 + 1 = 4
        # 5 total obs needed to fit, 6 to compare rmse
        p_start, d_start, q_start = (1,0,0)
        P_start, D_start, Q_start, m_start = (1,0,0,0)
        p,d,q = (2,1,1)
        P,D,Q,m = (2,1,1,1)
        order_arg_ranges = (range(p_start, p),
                            range(d_start, d),
                            range(q_start, q),
                            range(P_start, P),
                            range(D_start, D),
                            range(Q_start, Q),
                            range(m_start, m))

        with self.assertRaises(ValueError):
            model = BruteForceARIMARegression(y, X, order_arg_ranges)

    # Covers:
    #   len y = 2, # exog vars = 0
    #   p start = 0, p end = 1, p interval len = 1
    #   d start = 0, d end = 1, d interval len = 1
    #   q start = 0, q end = 1, q interval len = 1
    #   P start = 0, P end = 1, P interval len = 1
    #   D start = 0, D end = 1, D interval len = 1
    #   Q start = 0, Q end = 1, Q interval len = 1
    #   m start = 2, m end = 3, m interval len = 1
    #   OverspecifiedError = Raises, ValueError = None
    def test_simple_overspecified(self):
        n_obs = 2
        idx = pd.period_range(start=pd.Period(year=2003, month=2, freq='M'), periods=n_obs, freq='M')
        y = pd.Series(data=np.random.rand(n_obs),
                        index=idx,
                        dtype='float64')
        X = pd.DataFrame(
                        index=idx,
                        dtype='float64')
        p_start, d_start, q_start = (0,0,0)
        P_start, D_start, Q_start, m_start = (0,0,0,0)
        p,d,q = (1,1,1)
        P,D,Q,m = (1,1,1,1)
        order_arg_ranges = (range(p_start, p),
                            range(d_start, d),
                            range(q_start, q),
                            range(P_start, P),
                            range(D_start, D),
                            range(Q_start, Q),
                            range(m_start, m))

        with self.assertRaises(OverspecifiedError):
            model = BruteForceARIMARegression(y, X, order_arg_ranges)

# TODO test where picks between more than one option